import keras
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, ReLU, Dropout

# 建立 Sequential 模型
model = Sequential()

# 卷积层 32个卷积核 感知野5*5 填充格式为边缘补零 激活函数为relu 输入格式28*28*1
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(ReLU())

# 卷积层 32个卷积核 感知野3*3 填充格式为边缘补零 激活函数为relu
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(ReLU())

# 卷积层 32个卷积核 感知野5*5 步长为2 激活函数为relu
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2)))
model.add(BatchNormalization())
model.add(ReLU())

# 卷积层 64个卷积核 感知野3*3 填充格式为边缘补零 激活函数为relu
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(ReLU())

# 卷积层 64个卷积核 感知野3*3 填充格式为边缘补零 激活函数为relu
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(ReLU())

# 卷积层 64个卷积核 感知野5*5 步长为2 激活函数为relu
model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2)))
model.add(BatchNormalization())
model.add(ReLU())

# 全连接层，神经元个数128 激活函数为relu
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.4))

# 全连接层，神经元个数10 激活函数为softmax
model.add(Dense(10, activation='softmax'))

# 输出模型各层的参数状况
model.summary()

# 用adam优化器和交叉熵损失进行编译
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

keras.utils.plot_model(
        model,
        to_file='modelpytorch.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=900
    )
