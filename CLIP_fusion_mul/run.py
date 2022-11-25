# coding: UTF-8
import torch
from Model_parameter_setting import Config
from classfier import Classfier
from MultiData import DataSet
from multi_train_test import train
import clip

if __name__ == '__main__':
    device = torch.device("cuda")
    # 相应模型的配置
    model = Classfier().to(device)
    config=Config()
    #得到clip模型
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print("Loading data...")
    # 数据集
    train_data = DataSet('train',config,preprocess,clip_model)
    test_data = DataSet('test',config,preprocess,clip_model)

    # train
    train(config, model, train_data, test_data)

    
    
    
    