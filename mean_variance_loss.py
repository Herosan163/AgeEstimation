from torch import nn
import math
import torch
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
        # mean loss
        a = torch.arange(self.start_age, self.end_age + 1, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target)**2
        mean_loss = mse.mean() / 2.0

        # variance loss
        b = (a[None, :] - mean[:, None])**2
        variance_loss = (p * b).sum(1, keepdim=True).mean()
        
        return self.lambda_1 * mean_loss, self.lambda_2 * variance_loss
