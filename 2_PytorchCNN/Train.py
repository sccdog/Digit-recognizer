from CNN import train
from Submit import submit

NUM_EPOCHS = 200  # 训练轮数
BATCH_SIZE = 64  # 批大小
LEARNING_RATE = 0.001  # 学习率

result = train(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)

submit(result)
