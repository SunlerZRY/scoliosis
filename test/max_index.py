from model.utils import MaxIndex
import torch

a = torch.Tensor([[0, 0, 1, 0],
                  [0, 1, 1, 1],
                  [0, 0, 0, 0]])

r = MaxIndex(a, 3)
print(r, type(r))