import torch
import torch.nn as nn
import roast_layers

def test_local_1():
  layer = roast_layers.RLinear(
      in_features = 1024,
      out_features  = 2048, 
      roast_alpha = 1.0,
      roast_dropout = 0,
      roast_seed = 2023,
      hashed_weight = None,
      compression = 0.01,
      bias=False,
      ).cuda()
  print(layer)
  x = (2 * torch.rand(32, 1024) -1 ).cuda()
  y = layer(x)
  print(y.sum())
 

def test_global_1():
  hashed_weight = nn.Parameter(torch.rand(1024*10,)*2-1)
  layer = roast_layers.RLinear(
      in_features = 1024,
      out_features  = 2048, 
      roast_alpha = 1.0,
      roast_dropout = 0,
      roast_seed = 2023,
      hashed_weight = hashed_weight,
      compression = None,
      bias=False,
      ).cuda()

  print(layer)
  x = (2 * torch.rand(32, 1024) -1 ).cuda()
  y = layer(x)
  print(y.sum())

test_local_1() 
test_global_1() 
