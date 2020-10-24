


import torch
import numpy as np
import scipy.sparse as sp

from torch import nn
from torch import optim
import torch.nn.functional as F

from implicit.Block import LayerBlock
from implicit.Model import SequentialBlockModel
from implicit.Ops import FlattenOpL
from nn import ImplicitLayer, fgsm_test
from utils import transpose, mnist_load, get_valid_accuracy, get_robust_accuracy, Logger


class LeNet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        self.relu = torch.nn.ReLU()
        self.avgpool = torch.nn.AvgPool2d(2)

        self.flatten = FlattenOpL()
        self.block1 = LayerBlock([self.conv1], [self.relu])
        self.block2 = LayerBlock([self.avgpool, self.conv2], [self.relu])
        self.block3 = LayerBlock([self.avgpool, self.flatten, self.fc1], [self.relu])
        self.block4 = LayerBlock([self.fc2], [self.relu])
        self.block5 = LayerBlock([self.fc3], [])
        self.model = SequentialBlockModel([self.block1, self.block2, self.block3, self.block4, self.block5], populate_B=True, populate_C=True)

    def forward(self, xs):
        xs = xs.reshape((-1, 1, 28, 28))
        xs = torch.nn.functional.pad(xs, (2, 2, 2, 2), 'constant', 0)
        return self.model(xs)


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

epoch = 10
bs = 100
lr = 5e-3

train_ds, train_dl, valid_ds, valid_dl = mnist_load(bs)

model = LeNet5()
opt = optim.Adam(model.parameters(), lr=lr)
loss_fn = F.cross_entropy
model.to(device)

#logger = Logger(printstr=["batch: {}. loss: {:.2f}, valid_loss/acc: {:.2f}/{}, robust_loss/acc: {:.2f}/{}, FGSM_loss/acc: {:.2f}/{}", "batch", "loss", "valid_loss", "valid_acc", "robust_loss", "robust_acc", "FGSM_loss", "FGSM_acc"],
#                dir_name="NN_784_100_10")
#logger = Logger(printstr=["batch: {}. loss: {:.2f}, valid_loss/acc: {:.2f}/{}", "batch", "loss", "valid_loss", "valid_acc"],
#                dir_name="LeNet5_Implicit_Construction")
l = Logger.load_profile('/home/beeperman/Projects/ImplicitLayers/logs/LeNet5_Implicit_Construction_2019-11-20_14:32:45')
model.load_state_dict(l.data_dict[l.BEST], strict=False)
res = get_valid_accuracy(model, loss_fn, valid_dl, device)
print(res)

def implicit_forward(xs):
    xs = xs.reshape((-1, 1, 28, 28))
    xs = torch.nn.functional.pad(xs, (2, 2, 2, 2), 'constant', 0)
    A, B, C, D, phi = model.model.getImplicitModel(xs[0:1])
    i = SequentialBlockModel.implicit_forward(sp.csr_matrix(A), B, C, D, phi, xs)
    return i

res = get_valid_accuracy(implicit_forward, loss_fn, valid_dl, device)
print(res)

a = 1