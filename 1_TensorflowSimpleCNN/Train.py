from Dataloader import loaddata
# from DataAugmentation import dataaug
from DataAugmentation import nodataaug
from CNN import train
from Submit import submit

# 数据加载
train_train, train_val, label_train, label_val, test_data = loaddata()

# 数据增广
# dataauged = dataaug(train_train)
dataauged = nodataaug(train_train)

# 设置训练轮数
epochs = 50
# 设置批大小
batch_size = 86

# 训练
trained, trainedmodel = train(dataauged, train_train, label_train, batch_size, epochs,
                              train_val, label_val)

# 生成提交数据
submit(trainedmodel, test_data)
