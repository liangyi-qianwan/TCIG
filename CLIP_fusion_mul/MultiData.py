from CLIP_textencoder import text
import torch
from torch.utils.data import DataLoader
from optain_imagedata import MyDataSet
from get_features import features


device = torch.device("cuda")

def DataSet(path,config,preprocess,clip_model):
    label=[]
    data=[]
    print("开始获取标签")
    with open('text_data/'+path+"_label.txt", 'r', encoding='utf-8') as f0:
        for lin in f0.readlines():
            label.append(torch.tensor(int(lin.replace('\n',''))).to(device))
    print("开始获取文本特征和图片特征")
    text_path, image_path = text(path)
    print("开始计算")

    image_dataset = MyDataSet(images_path=image_path,
                              images_class=label,
                             preprocess=preprocess,
                             model=clip_model)

    
    imagedata_load = DataLoader(image_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            collate_fn=image_dataset.collate_fn,
                            drop_last=False
                            )

    textdata_load= DataLoader(text_path,
                            batch_size=config.batch_size,
                            shuffle=False,
                            drop_last=False
                            )

    text_features,image_features=features(imagedata_load,textdata_load,clip_model)
    
    for i in range(len(text_features)):
        data.append((text_features[i],image_features[i],label[i]))

    data_load= DataLoader(data,
                            batch_size=config.batch_size,
                            shuffle=False,
                            drop_last=False
                            )

    return data_load

