import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import datetime

class  VideoDemo():
    def __init__(self,
                 detector,
                 camera=0,
                 width=640,
                 height=480,
                 record=False,
                 filename="demo.mp4"):
        self.camera = camera
        self.detector = detector
        self.width = width
        self.height = height
        self.record = record
        self.filename = filename
        self.videowriter = None
        self.initialize()

    def initialize(self):
        self.capture = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            print("Error opening video camera")
            return

        # cap.set(cv2.CAP_PROP_FPS, 5)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.record:
            self.videowriter = cv2.VideoWriter(self.filename,
                                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                                10,
                                                (self.width, self.height), 
                                                isColor=True)

    def loop(self):
        font = cv2.FONT_HERSHEY_DUPLEX
        line_type = 1
        class_labels = {1:"Summit Drinking Water", 2:"Coke", 3:"Pineapple Juice"}
        colors = {1:(255, 0, 0), 2:(0, 0, 255), 3:(0,255,0)}
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        while True:
            start_time = datetime.datetime.now()
            ret, image = self.capture.read()
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = transform(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(device=device)
            predictions = self.detector(img)[0]
            boxes = predictions['boxes'].tolist()
            labels = predictions['labels'].tolist()
            
            for i in range(len(labels)):
                rect = boxes[i]
                x1 = rect[0]
                y1 = rect[1]
                x2 = rect[2]
                y2 = rect[3]
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                name = class_labels[labels[i]]
                index = labels[i]
                color = colors[index]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                cv2.putText(image,
                            name,
                            (x1, y1-15),
                            font,
                            0.5,
                            color,
                            line_type)

            cv2.imshow('image', image)
            if self.videowriter is not None:
                if self.videowriter.isOpened():
                    self.videowriter.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue
        self.capture.release()
        cv2.destroyAllWindows()

def detection_model(num_classes):
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 4
model = detection_model(num_classes)
model.load_state_dict(torch.load("model_epoch_4.pth", map_location=device))
model.eval()

videodemo = VideoDemo(detector=model, camera=0, record=False, filename="demo.mp4")
videodemo.loop()