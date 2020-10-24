import torch
import numpy as np
import scipy.sparse as sp
from implicit.Block import LayerBlock
from implicit.Model import SequentialBlockModel
from implicit.Ops import FlattenOp, FlattenOpL

input = torch.randn((10, 784))
input = input.reshape((10, 1, 28, 28))
input = torch.nn.functional.pad(input, (2,2,2,2), 'constant', 0)

conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
fc1 = torch.nn.Linear(16*5*5, 120)
fc2 = torch.nn.Linear(120, 84)
fc3 = torch.nn.Linear(84, 10)

maxpool = torch.nn.MaxPool2d(2)
avgpool = torch.nn.AvgPool2d(2)
relu = torch.nn.ReLU()
flatten = FlattenOp()

# simple test
block1 = LayerBlock([conv1], [relu, maxpool])
block2 = LayerBlock([conv2], [relu, maxpool, flatten])
block3 = LayerBlock([fc1], [relu])
block4 = LayerBlock([fc2], [relu])
block5 = LayerBlock([fc3], [])
model = SequentialBlockModel([block1, block2, block3, block4, block5])

#flatten = FlattenOpL()
#block1 = LayerBlock([conv1], [relu])
#block2 = LayerBlock([avgpool, conv2], [relu])
#block3 = LayerBlock([avgpool, flatten, fc1], [relu])
#block4 = LayerBlock([fc2], [relu])
#block5 = LayerBlock([fc3], [])
#model = SequentialBlockModel([block1, block2, block3, block4, block5])


# full model
tensor_out = model(input)
model.populate_B=1
model.populate_C=1
A, B, C, D, phi = model.getImplicitModel(input[0:1, :])
i = SequentialBlockModel.implicit_forward(A, B, C, D, phi, input)
t = tensor_out.detach().cpu().numpy()
assert np.isclose(np.abs(i - t).max(), 0, atol=1e-5)

import matplotlib
from matplotlib import pyplot as plt
plt.spy(sp.bmat([[A, B]]), markersize=0.01)
plt.show()



# block
#tensor_out = block1(input)
#i = block1.implicit_forward(input)
#t = tensor_out.detach().cpu().numpy().reshape(10, -1)
#assert np.isclose(np.abs(i - t).max(), 0, atol=1e-5)

