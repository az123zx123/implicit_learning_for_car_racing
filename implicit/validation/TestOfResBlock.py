import torch
import numpy as np
import scipy.sparse as sp
from implicit.Block import LayerBlock
from implicit.Model import SequentialBlockModel
from implicit.ResNetModel import ResNetModel, ResBasicBlock



input = torch.randn((10, 784*3))
input = input.reshape((10, 3, 28, 28))
input = torch.nn.functional.pad(input, (2,2,2,2), 'constant', 0)

conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
bn1 = torch.nn.BatchNorm2d(16)
relu1 = torch.nn.ReLU()
block1 = LayerBlock([conv1, bn1], [relu1])

block1.eval()
tensor_out = block1(input)
i = block1.implicit_forward(input)
t = tensor_out.detach().cpu().numpy().reshape([10, -1])
assert np.isclose(np.abs(i - t).max(), 0, atol=5e-5)

block2 = ResBasicBlock(16, 32, 2)
block2.eval()
input = tensor_out
tensor_out = block2(input)
i = block2.implicit_forward(input)
t = tensor_out.detach().cpu().numpy().reshape([10, -1])
assert np.isclose(np.abs(i - t).max(), 0, atol=5e-5)

block3 = ResBasicBlock(32, 64, 2)
block3.eval()
input = tensor_out
tensor_out = block3(input)
i = block3.implicit_forward(input)
t = tensor_out.detach().cpu().numpy().reshape([10, -1])
assert np.isclose(np.abs(i - t).max(), 0, atol=5e-5)


a=1
