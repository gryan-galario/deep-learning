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
model_pth = "model_epoch_4.pth"

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

if not(os.path.isfile(model_pth)):
    print("downloading trained model weights")
    id = "1-Bs82OZ7-CbQKZf1NI8XGUAELVUFSJUd"
    gdown.download(id=id, output=model_pth, quiet=False)

class TargetTransform(object):
  """Convert bbox and labels to expected format"""
  def __call__(self, sample):
    # output = []
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
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model

coco_annotation_test_file = "labels_test_converted.json"
transform = torchvision.transforms.ToTensor()
target_transform = TargetTransform()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 4
model = detection_model(num_classes)
model.load_state_dict(torch.load("model_epoch_4.pth", map_location=device))
model.to(device)

test_dataset = torchvision.datasets.CocoDetection(pth, coco_annotation_test_file, transform=transform, target_transform=target_transform)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

evaluate(model, data_loader_test, device=device)