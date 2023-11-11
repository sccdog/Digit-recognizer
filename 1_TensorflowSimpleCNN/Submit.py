import numpy as np
import pandas as pd
from keras.models import load_model


def submit(model, testdata):

    trainedmodel = model
    test = testdata

    # 识别数字 predict results
    results = trainedmodel.predict(test)

    # 返回最大值索引 select the index with the maximum probability
    results = np.argmax(results, axis=1)

    results = pd.Series(results, name="Label")

    submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

    submission.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    h5model = load_model('models/model_simpleCNN.h5')
    test_data = pd.read_csv('digit-recognizer/test.csv')
    test_data = test_data / 255.0
    test_data = test_data.values.reshape(-1, 28, 28, 1)
    results1 = h5model.predict(test_data)
    results1 = np.argmax(results1, axis=1)
    results1 = pd.Series(results1, name="Label")
    submission1 = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results1], axis=1)
    submission1.to_csv("submission.csv", index=False)
