import os
import pandas as pd
import pathlib
import json
from PIL import Image

pth = pathlib.Path(__file__).parent.resolve()
datafolder = "drinks"
pth = os.path.join(pth, datafolder)
os.chdir(pth)
filename = "labels_test.csv"
jsonfile = filename[:-4] + "_converted.json"

df = pd.read_csv(filename)
df['id'] = df['frame'].apply(lambda x: x[3:-4]).astype(int)
df['bbox_width'] = df['xmax'] - df['xmin']
df['bbox_height'] = df['ymax'] - df['ymin']
#df['img_width'] = pd.Series(dtype='int')
#df['img_height'] = pd.Series(dtype='int')

categories = [{"id": 1, "name": "Summit Drinking Water"}, {"id":2, "name": "Coke"}, {"id": 3, "name":"Pineapple Juice"}]
images = []
annotations = []

for index, row in df.iterrows():
    with Image.open(row['frame']) as img:
        img_tmpdict = {}
        annotation_tmpdict = {}
        img_width, img_height = img.size
        #df.at[index, 'img_width'] = img_width
        #df.at[index, 'img_height'] = img_height
        img_tmpdict['id'] = row['id']
        img_tmpdict['width'] = img_width
        img_tmpdict['height'] = img_height
        img_tmpdict['file_name'] = row['frame']
        annotation_tmpdict['id'] = index + 1
        annotation_tmpdict['image_id'] = row['id']
        annotation_tmpdict['category_id'] = row['class_id']
        annotation_tmpdict['segmentation'] = []
        annotation_tmpdict['bbox'] = [row['xmin'], row['ymin'], row['bbox_width'], row['bbox_height']]
        images.append(img_tmpdict)
        annotations.append(annotation_tmpdict)

data = {"images": images, "annotations": annotations, "categories": categories}

with open(jsonfile, 'w') as file:
    file.write(json.dumps(data, indent=4))
