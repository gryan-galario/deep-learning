import json
import os
import pathlib

from matplotlib.font_manager import json_load

pth = pathlib.Path(__file__).parent.resolve()
datafolder = "drinks_small"
pth = os.path.join(pth, datafolder)
os.chdir(pth)

with open('segmentation_test.json') as file:
    data = json.load(file)
    main_key = '_via_img_metadata'
    sub_key = '0010050.jpg98518'
    third_keys = data[main_key][sub_key].keys()
    third_values = data[main_key][sub_key].values()
    print("keys:", third_keys)
    print("values:", third_values)   

    