import torch
import torch.nn.functional as F

def RobustCrossEntropy(y_hat, sigma_y, y):
    if len(y.shape) >= 2:
        # TODO: deal with one-hot labels
        raise NotImplementedError()
    else:
        # y are indexes
        return F.cross_entropy(y_hat+sigma_y, y) + 2 * torch.mean(sigma_y[0, y])
