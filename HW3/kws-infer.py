F'''
Runs an inference on a single audio file.
Assumption is data file and checkpoint are in the same args.path
Simple test:
    python3 kws-infer.py --wav-file <path-to-wav-file>  
To use microphone input with GUI interface, run:
    python3 kws-infer.py --gui
    On RPi 4:
    python3 kws-infer.py --rpi --gui
Dependencies:
    sudo apt-get install libasound2-dev libportaudio2 
    pip3 install pysimplegui
    pip3 install sounddevice 
    pip3 install librosa
    pip3 install validators
Inference time:
    0.03 sec Quad Core Intel i7 2.3GHz
    0.08 sec on RPi 4
'''


import torch
from torch import nn
import argparse
import torchaudio
import os
import numpy as np
import librosa
import sounddevice as sd
import time
import validators
from torchvision.transforms import ToTensor
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import accuracy
from einops import rearrange

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/speech_commands/")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--wav-file", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="transformer-kws-best-acc-v4.ckpt")
    parser.add_argument("--gui", default=False, action="store_true")
    parser.add_argument("--rpi", default=False, action="store_true")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    return args


# main routine
if __name__ == "__main__":
    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    idx_to_class = {i: c for i, c in enumerate(CLASSES)}

    args = get_args()

    class Attention(nn.Module):
        def __init__(self, dim, num_heads=8, qkv_bias=False):
            super().__init__()
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
            head_dim = dim // num_heads
            self.scale = head_dim ** -0.5

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)

            return x
    
    class Mlp(nn.Module):
        """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
      
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
            return x
    
    class Block(nn.Module):
        def __init__(
                self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
            super().__init__()
            self.norm1 = norm_layer(dim)
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias) 
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer) 
    

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x
    
    class Transformer(nn.Module):
        def __init__(self, dim, num_heads, num_blocks, mlp_ratio=4., qkv_bias=False,  
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm):
            super().__init__()
            self.blocks = nn.ModuleList([Block(dim, num_heads, mlp_ratio, qkv_bias, 
                                        act_layer, norm_layer) for _ in range(num_blocks)])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x
    
    def init_weights_vit_timm(module: nn.Module):
        """ ViT weight initialization, original timm impl (for reproducibility) """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, 'init_weights'):
            module.init_weights()
    
    class LitTransformer(LightningModule):
        def __init__(self, num_classes=10, lr=0.001, max_epochs=30, depth=12, embed_dim=64,
                    head=4, patch_dim=192, seqlen=16, **kwargs):
            super().__init__()
            self.save_hyperparameters()
            self.encoder = Transformer(dim=embed_dim, num_heads=head, num_blocks=depth, mlp_ratio=4.,
                                    qkv_bias=False, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
            self.embed = torch.nn.Linear(patch_dim, embed_dim)

            self.fc = nn.Linear(seqlen * embed_dim, num_classes)
            self.loss = torch.nn.CrossEntropyLoss()
            
            self.reset_parameters()


        def reset_parameters(self):
            init_weights_vit_timm(self)
        

        def forward(self, x):
            # Linear projection
            x = self.embed(x)
                
            # Encoder
            x = self.encoder(x)
            x = x.flatten(start_dim=1)

            # Classification head
            x = self.fc(x)
            return x
        
        def configure_optimizers(self):
            optimizer = Adam(self.parameters(), lr=self.hparams.lr)
            # this decays the learning rate to 0 after max_epochs using cosine annealing
            scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
            return [optimizer], [scheduler]

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss(y_hat, y)
            return loss
        

        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss(y_hat, y)
            acc = accuracy(y_hat, y)
            return {"y_hat": y_hat, "test_loss": loss, "test_acc": acc}

        def test_epoch_end(self, outputs):
            avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
            avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
            self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
            self.log("test_acc", avg_acc*100., on_epoch=True, prog_bar=True)

        def validation_step(self, batch, batch_idx):
            return self.test_step(batch, batch_idx)

        def validation_epoch_end(self, outputs):
            return self.test_epoch_end(outputs)
    
    #model = LitTransformer(num_classes=37, lr=0.001, epochs=30, depth=20, embed_dim=64, head=8, patch_dim=512, seqlen=8,)

    if validators.url(args.checkpoint):
        checkpoint = args.checkpoint.rsplit('/', 1)[-1]
        # check if checkpoint file exists
        if not os.path.isfile(checkpoint):
            torch.hub.download_url_to_file(args.checkpoint, checkpoint)
    else:
        checkpoint = args.checkpoint

    print("Loading model checkpoint: ", checkpoint)
    #scripted_module = torch.jit.load(checkpoint)
    scripted_module = LitTransformer.load_from_checkpoint(checkpoint)

    if args.gui:
        import PySimpleGUI as sg
        sample_rate = 16000
        sd.default.samplerate = sample_rate
        sd.default.channels = 1
        sg.theme('DarkAmber')
  
    elif args.wav_file is None:
        # list wav files given a folder
        print("Searching for random kws wav file...")
        label = CLASSES[2:]
        label = np.random.choice(label)
        path = os.path.join(args.path, "SpeechCommands/speech_commands_v0.02/")
        path = os.path.join(path, label)
        wav_files = [os.path.join(path, f)
                     for f in os.listdir(path) if f.endswith('.wav')]
        # select random wav file
        wav_file = np.random.choice(wav_files)
    else:
        wav_file = args.wav_file
        label = args.wav_file.split("/")[-1].split(".")[0]

    if not args.gui:
        waveform, sample_rate = torchaudio.load(wav_file)

    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                     n_fft=args.n_fft,
                                                     win_length=args.win_length,
                                                     hop_length=args.hop_length,
                                                     n_mels=args.n_mels,
                                                     power=2.0)
    if not args.gui:
        mel = ToTensor()(librosa.power_to_db(transform(waveform).squeeze().numpy(), ref=np.max))
        mel = mel.unsqueeze(0)
        mels = rearrange(mel, 'b c h (p1 w) -> b p1 (c h w)', p1=8)

        pred = torch.argmax(scripted_module(mel), dim=1)
        print(f"Ground Truth: {label}, Prediction: {idx_to_class[pred.item()]}")
        exit(0)

    layout = [ 
        [sg.Text('Say it!', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 140), key='-OUTPUT-'),],
        [sg.Text('', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 100), key='-STATUS-'),],
        [sg.Text('Speed', expand_x=True, font=("Helvetica", 28), key='-TIME-')],
    ]

    window = sg.Window('KWS Inference', layout, location=(0,0), resizable=True).Finalize()
    window.Maximize()
    window.BringToFront()

    total_runtime = 0
    n_loops = 0
    while True:
        event, values = window.read(100)
        if event == sg.WIN_CLOSED:
            break
        
        waveform = sd.rec(sample_rate).squeeze()
        
        sd.wait()
        if waveform.max() > 1.0:
            continue
        start_time = time.time()
        if args.rpi:
            # this is a workaround for RPi 4
            # torch 1.11 requires a numpy >= 1.22.3 but librosa 0.9.1 requires == 1.21.5
            waveform = torch.FloatTensor(waveform.tolist())
            mel = np.array(transform(waveform).squeeze().tolist())
            mel = librosa.power_to_db(mel, ref=np.max).tolist()
            
            mel = torch.FloatTensor(mel)
            mel = mel.unsqueeze(0)

        else:
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            mel = ToTensor()(librosa.power_to_db(transform(waveform).squeeze().numpy(), ref=np.max))
        mel = mel.unsqueeze(0)
        mel = rearrange(mel, 'b c h (p1 w) -> b p1 (c h w)', p1=8)
        pred = scripted_module(mel)
        pred = torch.functional.F.softmax(pred, dim=1)
        max_prob =  pred.max()
        elapsed_time = time.time() - start_time
        total_runtime += elapsed_time
        n_loops += 1
        ave_pred_time = total_runtime / n_loops
        if max_prob > args.threshold:
            pred = torch.argmax(pred, dim=1)
            human_label = f"{idx_to_class[pred.item()]}"
            window['-OUTPUT-'].update(human_label)
            window['-OUTPUT-'].update(human_label)
            if human_label == "stop":
                window['-STATUS-'].update("Goodbye!")
                # refresh window
                window.refresh()
                time.sleep(1)
                break
                
        else:
            window['-OUTPUT-'].update("...")
        
        window['-TIME-'].update(f"{ave_pred_time:.2f} sec")


    window.close()
