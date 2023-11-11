import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

# 转换为独热编码
from keras.utils.np_utils import to_categorical

# 设置显示样式的参数
sns.set(style='white', context='notebook', palette='deep')


def loaddata():

    train_data = pd.read_csv('digit-recognizer/train.csv')
    test_data = pd.read_csv('digit-recognizer/test.csv')

    print(f'训练数据shape: {train_data.shape}')
    print(f'测试数据shape: {test_data.shape}')

    # 分开标签和像素数值
    train_label = train_data["label"]
    train_data = train_data.drop(labels=["label"], axis=1)

    # 归一化至[0,1]
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # Reshape三维图像(数量, height = 28px, width = 28px , canal = 1)
    train_data = train_data.values.reshape(-1, 28, 28, 1)
    test_data = test_data.values.reshape(-1, 28, 28, 1)

    # 将标签编码为一个独热向量 (例如: 2 -> [0,0,1,0,0,0,0,0,0,0])
    train_label = to_categorical(train_label, num_classes=10)

    # 设置随机种子
    random_seed = 2

    # 分割出训练集和验证集
    train_data_train, train_data_validate, train_label_train, train_label_validate = (
        train_test_split(train_data, train_label, test_size=0.1, random_state=random_seed))

    print(train_data_train.shape)

    return train_data_train, train_data_validate, train_label_train, train_label_validate, test_data
