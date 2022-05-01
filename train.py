import os
import gdown
import tarfile
import torch
import torchvision
from engine import train_one_epoch, evaluate
import utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

cwd = os.getcwd()
pth = os.path.join(cwd, "drinks")

if not(os.path.isdir(os.path.join(cwd, "drinks"))):
    print("drinks dataset folder not found")
    if (os.path.isfile("drinks.tar.gz")):
        print("extracting drinks.tar.gz")
        tar = tarfile.open("drinks.tar.gz", "r:gz")
        tar.extractall()
        tar.close()
        print("finished extracting")
    else:
        print("drinks.tar.gz not found")
        print("downloading drinks.tar.gz")
        id = "1AdMbVK110IKLG7wJKhga2N2fitV1bVPA"
        output = "drinks.tar.gz"
        gdown.download(id=id, output=output, quiet=False)
        print("extracting drinks.tar.gz")
        tar = tarfile.open("drinks.tar.gz", "r:gz")
        tar.extractall()
        tar.close()
        print("finished extracting")

class TargetTransform(object):
  """Convert bbox and labels to expected format"""
  def __call__(self, sample):
    boxes = []
    labels = []
    image_ids = []
    iscrowds = []
    areas = []
    if (type(sample) == list):
      for entry in sample:
        bbox = entry['bbox'] 
        category = entry['category_id']
        image_id = entry['image_id']
        iscrowd = entry['iscrowd']
        area = entry['area']
        # swap entries
        # json format: xmin, ymin, width, height
        # required format, xmin, ymin, xmax, ymax
        bbox_converted = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
        boxes.append(bbox_converted)
        labels.append(category)
        image_ids.append(image_id)
        areas.append(area)
        iscrowds.append(iscrowd)
    boxes = torch.FloatTensor(boxes)
    labels = torch.tensor(labels, dtype=torch.int64)
    image_ids = torch.tensor(image_ids, dtype=torch.int64)
    image_ids = torch.unique(image_ids)
    iscrowds = torch.tensor(iscrowd, dtype=torch.int64)
    areas = torch.tensor(areas)
    return {'boxes':boxes, 'labels':labels, 'image_id':image_ids, 'iscrowd':iscrowds, 'area':areas}

def detection_model(num_classes):
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model

coco_annotation_train_file = "labels_train_converted.json"
coco_annotation_test_file = "labels_test_converted.json"

transform = torchvision.transforms.ToTensor()
target_transform = TargetTransform()

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# bg + water + coke + pineabble
num_classes = 4
# use our dataset and defined transformations
train_dataset = torchvision.datasets.CocoDetection(pth, coco_annotation_train_file, transform=transform, target_transform=target_transform)
test_dataset = torchvision.datasets.CocoDetection(pth, coco_annotation_test_file, transform=transform, target_transform=target_transform)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

model = detection_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# train for 5 epochs
num_epochs = 5

for epoch in range(num_epochs):
  # train for one epoch, printing every 10 iterations
  x = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20)
  # update the learning rate
  lr_scheduler.step()
  # evaluate on the test dataset
  evaluate(model, data_loader_test, device=device)
  torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")