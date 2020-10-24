import torch
import numpy as np

from torch import nn
from torch import optim
import torch.nn.functional as F

from implicit.Block import LayerBlock
from implicit.Model import SequentialBlockModel
from implicit.Ops import FlattenOpL
from nn import ImplicitLayer, fgsm_test
from utils import transpose, mnist_load, get_valid_accuracy, get_robust_accuracy, Logger, cifar_load


class LeNet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
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
        self.model = SequentialBlockModel([self.block1, self.block2, self.block3, self.block4, self.block5])

    def forward(self, xs):
        xs = xs.reshape((-1,3,32,32))
        return self.model(xs)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

epoch = 10
bs = 100
lr = 5e-3

train_ds, train_dl, valid_ds, valid_dl = cifar_load(bs)

model = LeNet5()
opt = optim.Adam(model.parameters(), lr=lr)
loss_fn = F.cross_entropy
model.to(device)

#logger = Logger(printstr=["batch: {}. loss: {:.2f}, valid_loss/acc: {:.2f}/{}, robust_loss/acc: {:.2f}/{}, FGSM_loss/acc: {:.2f}/{}", "batch", "loss", "valid_loss", "valid_acc", "robust_loss", "robust_acc", "FGSM_loss", "FGSM_acc"],
#                dir_name="NN_784_100_10")
logger = Logger(printstr=["batch: {}. loss: {:.2f}, valid_loss/acc: {:.2f}/{}", "batch", "loss", "valid_loss", "valid_acc"],
                dir_name="CIFAR10_LeNet5_Implicit_Construction")

for i in range(epoch):
    j = 0
    for xs, ys in train_dl:

        pred = model(xs.to(device))
        loss = loss_fn(pred, ys.to(device))

        loss.backward()
        opt.step()
        opt.zero_grad()
        valid_res = get_valid_accuracy(model, loss_fn, valid_dl, device)
        #robust_res = get_robust_accuracy(model, F.cross_entropy, 0.3, valid_dl, device)
        #FGSM_res = fgsm_test(model, device, test_loader=valid_dl, epsilon=0.1, do_print=False)

        # logging
        log_dict = {
            "batch": j,
            "loss": loss,
            "valid_loss": valid_res[0],
            "valid_acc":valid_res[1],
            #"robust_loss": robust_res[0],
            #"robust_acc": robust_res[1],
            #"FGSM_loss": FGSM_res[0],
            #"FGSM_acc": FGSM_res[1]
        }
        logger.log(log_dict, model, "valid_acc")
        j+=1
    print("--------------epoch: {}. loss: {}".format(i, loss))

a = 1