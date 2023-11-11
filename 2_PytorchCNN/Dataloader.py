import numpy as np  # 数据处理 Data processing
import pandas as pd  # 数据处理, CSV文件 I/O (例如 pd.read_csv)
from torch.utils.data import Dataset  # 数据载入 Data loading
from torch.utils.data.sampler import SubsetRandomSampler  # 数据预处理 Data preprocessing

train_set = pd.read_csv('digit-recognizer/train.csv')  # 读取训练数据 Load training data
test_set = pd.read_csv('digit-recognizer/test.csv')  # 读取测试数据 Load testing data

VALID_SIZE = 0.1  # 用于分割验证集的数据比例 The proportion of validation data

num_train = len(train_set)  # 训练集数据个数 42000 The number of train data
indices = list(range(num_train))  # 生成一列序号，从0-41999
np.random.shuffle(indices)  # 随机排列序号
split = int(np.floor(VALID_SIZE * num_train))  # 验证集数量 4200
train_indices, valid_indices = indices[split:], indices[:split]  # 分割验证集

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)

print(f'training set length: {len(train_indices)}')  # 37800
print(f'validation set length: {len(valid_indices)}')  # 4200


class DatasetMNIST(Dataset):
    def __init__(self, data, transform=None, labeled=True):
        self.data = data
        self.transform = transform
        self.labeled = labeled

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  # 重写，定义数据载入行为
        item = self.data.iloc[index]
        if self.labeled:  # 处理已标注数据
            x = item[1:].values.astype(np.uint8).reshape((28, 28))  # 图像像素数据
            y = item[0]  # 标注数字
        else:  # 处理未标注数据（缺少标注列，dimension不同，本文选择分别处理）
            x = item[0:].values.astype(np.uint8).reshape((28, 28))  # 图像像素数据
            y = 0  # 仅用于占位，数值不会被使用也不影响行为

        if self.transform is not None:
            x = self.transform(x)

        return x, y
