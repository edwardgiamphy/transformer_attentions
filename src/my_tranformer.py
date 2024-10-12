import torch
import torch.nn as nn
import torch.nn.functional as F

import typing
from typing import Optional, List

class Attention(nn.Module):
    """
    Implementation of attention module.
    Inputs:
        - TBD
    Outputs:
        - TBD
    """
    def __init__(self, d_model: int = 0, max_len: int = 0):
        """ 
        Initialize the transformer class.
        Inputs:
            - TBD
        Outputs:
            - TBD
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.W_q = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.W_k = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.W_v = nn.Linear(in_features=self.d_model, out_features=self.d_model)

        self.mask = self.create_mask(self.d_model, self.max_len)
    
    def create_mask(self, d_model: int = 0, max_len: int = 0) -> Optional[torch.Tensor]:
        """ 
        Creating a mask tensor for masked self-attention.
        Inputs:
            - TBD
        Outputs:
            - TBD
        """
        mask = torch.tril(torch.ones(max_len, max_len))
        mask[mask == 0.] = -torch.inf
        mask[mask == 1.] = 0.
        return mask
    
    def forward(self, encoded_q: Optional[torch.Tensor] = None, 
                encoded_k: Optional[torch.Tensor] = None, 
                encoded_v: Optional[torch.Tensor] = None, 
                parameter_mask: bool = False) -> Optional[torch.Tensor]:
        """ 
        Define the different steps in the transformer
        Inputs:
            - TBD
        Outputs:
            - TBD
        """
        q = self.W_q(encoded_q)
        k = self.W_k(encoded_k)
        v = self.W_v(encoded_v)

        values = torch.matmul(q, k.transpose(-1, -2))
        if parameter_mask :
            values += self.mask
        
        scaled_values = values / torch.sqrt(torch.tensor(self.d_model))
        attention = F.softmax(scaled_values, dim=-1)
        attention_values = torch.matmul(attention, v)
        return attention_values