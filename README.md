# drinks-object-detection
by Gryan Carl Galario

A trained object detection model on the [drinks dataset](https://drive.google.com/file/d/1AdMbVK110IKLG7wJKhga2N2fitV1bVPA/view)

This model uses [Faster R-CNN](https://arxiv.org/abs/1506.01497) as the base model for the object detection and a ResNet50 backbone for the image classification

## Data Preparation
1. Download and extract the drinks dataset. `convert_to_coco.py` is expected to be outside the drinks folder.
2. Manually change the filename in the `convert_to_coco.py` to convert the labels_train.csv or labels_test.csv in the drinks folder.
3. Run the following command on the terminal: `python convert_to_coco.py`
4. The converted annotation files shall then be moved outside the drinks folder to be used by the `train.py` and the `test.py`

## Install Requirements
`pip install -r requirements.txt`

## Testing the pretrained model
1. Run the following command: `python test.py`

## Training the model
1. Run the following command: `python train.py`
2. By default, the model is trained for 5 epochs. The training epoch can be manually changed by editing the py file itself.

## References
1. The `coco_eval.py`, `coco_utils.py`, `engine.py`, `utils.py`, and `transforms.py` were obtained from [pytorch vision reference](https://github.com/pytorch/vision/tree/main/references/detection)
2. The `demo.py` borrows heavily from [Dr. Atienza's code](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter11-detection/video_demo.py) but was modified to have the input and output processing compatible to the model
3. The `train.py` and `test.py` borrows heavily from the [pytorch object detection tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)


## Notes
Due to my lack of a GPU, I trained and tested the model in Google Colab. Here are the files I used:
* [Training notebook](https://colab.research.google.com/drive/1kEKffqCYVuKHlR3C1Y0mDLJZEDNJuE6P?usp=sharing)
* [Testing notebook](https://colab.research.google.com/drive/1v46bFWjRNPSX84qa340mBSw7bORQl2Z2?usp=sharing)
* [Demo notebook](https://colab.research.google.com/drive/1nbdqPnve1-7z0FVZY_XrgZdxFuPpM-qN?usp=sharing)
* [Weights checkpoint](https://drive.google.com/file/d/1-Bs82OZ7-CbQKZf1NI8XGUAELVUFSJUd/view?usp=sharing)

