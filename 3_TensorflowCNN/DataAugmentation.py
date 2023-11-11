from keras.preprocessing.image import ImageDataGenerator


def dataaug(train_data_train):
    # 增加数据以防止过拟合
    dataaugmented = ImageDataGenerator(
        featurewise_center=False,  # 在数据集上将输入平均值设置为0
        samplewise_center=False,  # 将每个样本的平均值设置为0
        featurewise_std_normalization=False,  # 将输入除以数据集的std
        samplewise_std_normalization=False,  # 将每个输入除以它的std
        zca_whitening=False,  # 使用ZCA白化
        rotation_range=10,  # 在范围内随机旋转图像（0到180度）
        zoom_range=0.1,  # 随机缩放图像
        width_shift_range=0.1,  # 水平随机移动图像（总宽度的一部分）
        height_shift_range=0.1,  # 垂直随机移动图像（总高度的一部分）
        horizontal_flip=False,  # 随机翻转图像
        vertical_flip=False)  # 随机翻转图像

    dataaugmented.fit(train_data_train)
    return dataaugmented
