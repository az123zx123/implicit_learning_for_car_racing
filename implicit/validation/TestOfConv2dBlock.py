import torch
import numpy as np
from implicit.Block import LayerBlock

## TODO: To be implemented

lin = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=1)
act = torch.nn.ReLU()
act1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
input = torch.randn((10, 30, 30, 1))

# simple test
block = LayerBlock([lin], [act])

tensor_out = block(input)
i = block.implicit_forward(input)
t = tensor_out.detach().cpu().numpy().reshape(10, -1)
assert np.isclose(np.abs(i - t).max(), 0, atol=1e-5)


# MaxPool Test
block = LayerBlock([lin], [act1])

tensor_out = block(input)
i = block.implicit_forward(input)
t = tensor_out.detach().cpu().numpy().reshape(10, -1)
t = t[:, block.out_size()]
assert np.isclose(np.abs(i - t).max(), 0, atol=1e-5)


# MaxPool Test
block = LayerBlock([lin], [act, act1])

tensor_out = block(input)
i = block.implicit_forward(input)
t = tensor_out.detach().cpu().numpy()
assert np.isclose(np.abs(i - t).max(), 0, atol=1e-5)
