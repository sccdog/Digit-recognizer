import torch
import pandas as pd
import numpy as np
from Dataloader import test_set
from CNN import test_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def submit(model):
    with torch.no_grad():
        model.eval()
        results = torch.ShortTensor()
        for predict_images, _ in test_dataloader:
            predict_images = predict_images.reshape(-1, 1, 28, 28).to(device)
            predict_outputs = model(predict_images)
            test_predictions = predict_outputs.cpu().data.max(1, keepdim=True)[1]
            results = torch.cat((results, test_predictions), dim=0)

        submission = pd.DataFrame(np.c_[np.arange(1, len(test_set) + 1)[:, None], results.numpy()],
                                  columns=['ImageId', 'Label'])
        submission.to_csv('submission.csv', index=False)
