import torch
import numpy as np
import scipy.sparse as sp
from implicit.Block import LayerBlock
from implicit.Model import SequentialBlockModel
from implicit.ResNetModel import ResNetModel



input = torch.randn((10, 784*3))
input = input.reshape((10, 3, 28, 28))
input = torch.nn.functional.pad(input, (2,2,2,2), 'constant', 0)


model = ResNetModel([3,3,3])
model.eval()
#model.seqmodel.blocklist = model.seqmodel.blocklist[:5]
tensor_out = model(input)

#out = input
#for block in model.seqmodel.blocklist:
#    out = block.implicit_forward(out)

A, B, C, D, phi = model.getImplicitModel(input[0:1, :])
i = SequentialBlockModel.implicit_forward(sp.csr_matrix(A), B, C, D, phi, input)
t = tensor_out.detach().cpu().numpy().reshape([10, -1])
assert np.isclose(np.abs(i - t).max(), 0, atol=np.abs(np.mean(i))*1e-3)
# check the padding part!

import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
plt.spy(sp.bmat([[A, B]]), markersize=0.003)
plt.show()

a = 1