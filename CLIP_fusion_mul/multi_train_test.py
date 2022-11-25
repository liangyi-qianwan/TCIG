# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import torch.optim as optim
import clip
from PIL import Image

device = torch.device("cuda")
def train(config, model,train_data,test_data):
    # 开始时间
    start_time = time.time()
    # 模型训练
    model.train()
    # 列出最有参数
    param_optimizer = list(model.named_parameters())
    # 偏置
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # item记录有几次loss每下降
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = optim.Adam(optimizer_grouped_parameters,
                         lr=config.learning_rate)
    total_batch = 0  # 记录进行到多少batch
    dev_best_acc = 0.000
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # 遍历每一行数据
        for train_text, train_image,labels in train_data:
            outputs = model(train_text.to(device).float(),train_image.to(device).float())
            # 清除所有优化的torch.Tensor的梯度
            model.zero_grad()
            # 损失  交叉损失函数
            loss = F.cross_entropy(outputs, labels)
            # 计算
            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, test_data)
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    # 保存模型
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%} {5}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, improve))
                model.train()
            total_batch += 1
        #     if total_batch - last_improve > config.require_improvement:
        #         # 验证集loss超过1000batch没下降，结束训练
        #         # print("No optimization for a long time, auto-stopping...")
        #         flag = True
        #         break
        # if flag:
        #     break
    test(config, model, test_data)

def test(config, model, test_data):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_data, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)


def evaluate(config, model, data, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts,images,labels in data:
            outputs = model(texts.to(device).float(),images.to(device).float())
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data), report, confusion
    return acc, loss_total / len(data)

