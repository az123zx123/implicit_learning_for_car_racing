import torch
import numpy as np
from implicit.Block import LayerBlock
from implicit.Model import SequentialBlockModel
import scipy.sparse as sp




lin = torch.nn.Linear(784,1024)
lin0 = torch.nn.Linear(1024,1024)
lin1 = torch.nn.Linear(1024,1024)
lin2 = torch.nn.Linear(1024,300)
lin3 = torch.nn.Linear(300,10)
act = torch.nn.ReLU()
input = torch.randn((10, 784))

# simple test
block = LayerBlock([lin], [act])
block0 = LayerBlock([lin0], [act])
block1 = LayerBlock([lin1], [act])
block2 = LayerBlock([lin2], [act])
block3 = LayerBlock([lin3], [])
model = SequentialBlockModel([block, block0, block1, block2, block3])

tensor_out = model(input)
A, B, C, D, phi = model.getImplicitModel(input[0:1, :])
i = SequentialBlockModel.implicit_forward(A, B, C, D, phi, input)
t = tensor_out.detach().cpu().numpy()
assert np.isclose(np.abs(i - t).max(), 0, atol=1e-5)

import matplotlib
from matplotlib import pyplot as plt
plt.spy(sp.bmat([[A, B]]), markersize=0.01)
plt.show()

# complicated
tensor_out = block(input)
i = block.implicit_forward(input)
t = tensor_out.detach().cpu().numpy()
assert np.isclose(np.abs(i - t).max(), 0, atol=1e-5)
