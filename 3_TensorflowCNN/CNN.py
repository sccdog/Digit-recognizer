import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, BatchNormalization
# from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau


def cnn():
    # 建立 Sequential 模型。Sequential 是 Keras 中的一种神经网络框架，可以被认为是一个容器。
    # 其中封装了神经网络的结构。Sequential 模型只有一组输入和一组输出。
    # 各层之间按照先后顺序进行堆叠。前面一层的输出就是后面一次的输入。
    # 通过不同层的堆叠，可以构建出神经网络。
    model = Sequential()

    # 第一卷积层 32个卷积核 感知野3*3 激活函数为relu 输入格式28*28*1
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     activation='relu', input_shape=(28, 28, 1)))

    # 批标准化层 将每层网络的输入的数据分布变成正态分布，有利于网络的稳定性，加快收敛。
    model.add(BatchNormalization())

    # 第二卷积层 32个卷积核 感知野3*3 激活函数为relu
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     activation='relu'))

    # 批标准化层
    model.add(BatchNormalization())

    # 第三卷积层 32个卷积核 感知野5*5 填充格式为边缘补零 激活函数为relu
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                     activation='relu'))
    # 批标准化层
    model.add(BatchNormalization())

    # Dropout正则化 在每轮权重更新时随机选择一定比例的节点抛弃
    model.add(Dropout(0.4))

    # 第四卷积层 64个卷积核 感知野3*3 激活函数为relu
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     activation='relu'))

    model.add(BatchNormalization())

    # 第五卷积层 64个卷积核 感知野3*3 激活函数为relu
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     activation='relu'))

    model.add(BatchNormalization())

    # 第六卷积层 64个卷积核 感知野5*5 填充格式为边缘补零 激活函数为relu
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Same',
                     activation='relu'))

    model.add(BatchNormalization())

    # Dropout正则化 在每轮权重更新时随机选择一定比例的节点抛弃
    model.add(Dropout(0.4))

    # 第七卷积层 128个卷积核 感知野4*4 填充格式为边缘补零 激活函数为relu
    model.add(Conv2D(filters=128, kernel_size=(4, 4),
                     activation='relu'))
    # 批标准化层
    model.add(BatchNormalization())

    # 扁平化层 将多维输入展成一维
    model.add(Flatten())

    # Dropout正则化 在每轮权重更新时随机选择一定比例的节点抛弃
    model.add(Dropout(0.4))

    # 全连接层 神经单元节点10个 激活函数softmax
    model.add(Dense(10, activation="softmax"))

    # 输出模型各层的参数状况
    model.summary()

    # 用adam优化器和交叉熵损失进行编译
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def train(dataauged, train_train, label_train, batch_size, epochs, train_val, label_val):
    model = cnn()
    # 设置学习率衰减
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=3,  # 多少个epoch没有改善之后更新学习率
                                                verbose=1,  # 每次更新的时候是否打印信息
                                                factor=0.3,  # 学习率降低的速率
                                                min_lr=0.00001)  # 最小学习率

    training = model.fit(dataauged.flow(train_train, label_train, batch_size=batch_size),
                         epochs=epochs, validation_data=(train_val, label_val),
                         verbose=2, steps_per_epoch=train_train.shape[0] // batch_size,
                         callbacks=[learning_rate_reduction])

    # 保存模型为h5文件，包括模型的结构、权重、训练配置（损失函数、优化器等）、优化器的状态（以便从上次训练中断的地方开始）
    save_path = r'models\model_TensorflowCNN.h5'
    model.save(save_path)

    return training, model
