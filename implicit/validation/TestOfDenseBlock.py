import torch
import numpy as np
from implicit.Block import LayerBlock
from implicit.Model import SequentialBlockModel





lin = torch.nn.Linear(100,20)
lin0 = torch.nn.Linear(100, 100)
lin1 = torch.nn.Linear(20, 20)
act = torch.nn.ReLU()
input = torch.randn((10, 100))

# simple test
block = LayerBlock([lin], [act])

tensor_out = block(input)
i = block.implicit_forward(input)
t = tensor_out.detach().cpu().numpy()
assert np.isclose(np.abs(i - t).max(), 0, atol=1e-5)


# complicated
block = LayerBlock([lin0, lin, lin1], [act, act])

tensor_out = block(input)
i = block.implicit_forward(input)
t = tensor_out.detach().cpu().numpy()
assert np.isclose(np.abs(i - t).max(), 0, atol=1e-5)
