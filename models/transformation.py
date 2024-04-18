import torch
import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self, dim_z, oneDirection=True, vocab_size=1000):
        super().__init__()

        self.w = None
        self.oneDirection = oneDirection

        if self.oneDirection:
            self.w = nn.Parameter(torch.randn(1, dim_z))
        else:
            self.w = nn.Linear(vocab_size, dim_z)

    def forward(self, z, y, step_sizes):

        interim = None
        if not self.oneDirection:
            assert y is not None
            interim = step_sizes * self.w(y)

        elif y is not None:
            assert len(y) == z.shape[0]
            interim = step_sizes * self.w

        z_transformed = z + interim
        z_transformed = z.norm() * z_transformed / z_transformed.norm()

        return z_transformed
