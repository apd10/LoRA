#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from rz_linear import RzLinear
import pdb
from math import sqrt


class RLinear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        roast_alpha: float,
        roast_dropout: float,
        roast_seed: int,
        hashed_weight: Optional[torch.Tensor] = None,
        compression: Optional[float] = None,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        if hashed_weight is None:
            assert(compression is not None)
            num_weight_params = int(in_features * out_features * compression)
            hashed_weight = nn.Parameter(torch.zeros(num_weight_params, dtype=torch.float32), requires_grad=True)

        init_factor = sqrt(1. / in_features)
        self.hashed_weight = hashed_weight
        self.init_factor = init_factor
        self.roast_alpha = roast_alpha
        self.roast_seed = roast_seed
        if roast_dropout > 0.:
            self.roast_dropout = nn.Dropout(p = roast_dropout)
        else:
            self.roast_dropout = lambda x : x
        self.delta_roast_linear = RzLinear(in_features, out_features, 32 # not used
                                          ,self.hashed_weight, seed=roast_seed, init_factor = init_factor)
        
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'hashed_weight'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.hashed_weight)

    def forward(self, x: torch.Tensor):
        ''' the evaluation right now just does this. but can be improved to store merged weight '''
        result1 = F.linear(x, self.weight, bias=self.bias)
        result2 = self.roast_alpha * self.delta_roast_linear(self.roast_dropout(x))
        result = result1 + result2
        return result


