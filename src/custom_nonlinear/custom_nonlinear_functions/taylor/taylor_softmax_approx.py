import torch
import math
from custom_nonlinear.custom_approx import CustomSoftmax
import os

# Code functions with fp16 precision as input / output, and is not tested for other datatypes.
# Edit a to adjust degree center
# Edit degrees to adjust number of polynomial degrees

class TaylorSoftmax(CustomSoftmax):
    def __init__(self, layer, device, profile_path, profile_dims, profile=False):
        super(TaylorSoftmax, self).__init__(layer, device, profile_path, profile_dims, profile)

    def set_params(self, degree_center, degrees, config_path):
        self.config_path = config_path
        self.degree_center = torch.tensor(degree_center)
        self.degrees = degrees
        self.build_taylor()

    def build_taylor(self):
        prev_exp = torch.tensor(1).to(torch.float16)
        x_neg = torch.tensor(0).to(torch.float64)
        exp = torch.tensor(0).to(torch.float64)
        while True:
            for i in range(self.degrees):
                i = torch.tensor(i, dtype=torch.int64)
                intermediate = x_neg - self.degree_center
                intermediate.pow_(i)
                inv_fac = torch.exp(-torch.lgamma(torch.tensor(i + 1)))
                intermediate.mul_(inv_fac)
                exp.add_(intermediate)
            exp.mul_(torch.exp(self.degree_center))
            if exp <= 0.001 or prev_exp < exp or torch.isinf(exp):
                break
            else:
                prev_exp = exp.clone()
                x_neg.sub_(0.25)
        self.x_neg = x_neg.to(torch.bfloat16)

    def taylor_exp(self, x):
        self.x_neg = self.x_neg.to(self.device)
        exp = torch.zeros_like(x, dtype=torch.bfloat16)
        
        for i in range(self.degrees + 1):
            intermediate = x - self.degree_center
            intermediate.pow_(i)
            inv_fac = torch.exp(-torch.lgamma(torch.tensor(i + 1)))
            intermediate.mul_(inv_fac)
            exp.add_(intermediate)
        del inv_fac
        del intermediate

        zero_mask = x < self.x_neg
        del x

        exp_pow = torch.exp(self.degree_center)
        exp.mul_(exp_pow)
        exp[zero_mask] = 0

        return exp
    
    def nonlinear(self, attn_weights, dim=-1, dtype=torch.bfloat16):
        attn_weights = attn_weights.to(torch.bfloat16)
        attn_weights_max = torch.max(attn_weights, dim = dim, keepdim = True)[0]
        attn_weights = attn_weights - attn_weights_max
        del attn_weights_max

        attn_weights = self.taylor_exp(attn_weights)

        attn_weights_sum = torch.sum(attn_weights, dim = dim, keepdim = True)
        attn_weights.div_(attn_weights_sum)
        return attn_weights