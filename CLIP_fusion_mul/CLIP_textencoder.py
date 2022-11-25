import torch
from PIL import Image
device = torch.device("cuda")

def text(cls):
    text_path=[]
    image_path=[]
    with open("text_data//"+cls+"_data.txt", "r", encoding="utf-8") as file:
        item=0
        for line in file.readlines():
            image = "image_data//"+cls+"_image//{}.png".format(item)
            text_path.append(line.replace('\n',''))
            image_path.append(image)
            item+=1
    return text_path, image_path




