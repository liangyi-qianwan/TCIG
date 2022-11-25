import torch



class Config(object):

    """配置参数"""
    def __init__(self):
        self.class_list = [x.strip() for x in open(
            'text_data/class.txt').readlines()]                                # 类别名单
        self.save_path = 'weight/text.ckpt'        # 模型训练结果
        self.device = torch.device('cuda')   # 设备

        self.require_improvement = 1000                                 # 若超过10000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 100                                             # epoch数
        self.batch_size = 20                                           # mini-batch大小
        self.learning_rate = 5e-5                                    # 学习率



