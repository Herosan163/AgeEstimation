from torch import nn
import math
import torch
# import numpy as np
import torch.nn.functional as F


class MeanVarianceLoss(nn.Module):

    def __init__(self, lambda_1, lambda_2, start_age, end_age):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start_age = start_age
        self.end_age = end_age

    def forward(self, input, target):

        N = input.size()[0]
        target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=1)
        p = m(input)
        # print('input', input.size())
        # print('target', target.size())
        # print('p', p.size())
        # mean loss
        a = torch.arange(self.start_age, self.end_age + 1, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target)**2
        mean_loss = mse.mean() / 2.0
        # print('mean', mean.size())
        # print('mse', mse.size())
        # print('mean_loss', mean_loss)

        # variance loss
        b = (a[None, :] - mean[:, None])**2
        variance_loss = (p * b).sum(1, keepdim=True).mean()
        # print(variance_loss)
        
        # return self.lambda_1 * mean_loss, torch.tensor(0.)
        return self.lambda_1 * mean_loss, self.lambda_2 * variance_loss
