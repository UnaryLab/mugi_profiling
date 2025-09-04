import torch
from custom_nonlinear.custom_approx import CustomGelu
import os

# Code functions with fp16 precision as input / output, and is not tested for other datatypes.
# Edit exp_dim to adjust the LUT size
# Edit max exp to adjust the maximum exponent of the LUT
class VLPGelu(CustomGelu):
    def __init__(self, layer, device, profile_path, profile_dims, profile=False):
        super(VLPGelu, self).__init__(layer, device, profile_path, profile_dims, profile)

    def set_params(self, exp_dim, mant_dim, window_size, lut_build, max_exp=None, min_exp=None, keys=None):
        self.keys = keys
        self.lut_build = lut_build
        self.exp_dim = exp_dim
        self.mant_dim = mant_dim
        self.window_size = window_size

        if lut_build == 'max':
            self.max_exp = max_exp
            self.min_exp = max_exp - (exp_dim - 1)
        elif lut_build == 'min':
            self.min_exp = min_exp
            self.max_exp = min_exp + (exp_dim - 1)
        else:
            raise ValueError("lut_build must be 'max' or 'min'")

    def window_gelu_approx(self, exp, mant, positive, negative):
        input_shape = exp.shape

        exp = exp.reshape(-1, input_shape[-1])
        mant = mant.reshape(-1, input_shape[-1])
        positive = positive.reshape(-1, input_shape[-1])
        negative = negative.reshape(-1, input_shape[-1])

        inter_shape = exp.shape

        if exp.shape[-1] % self.window_size != 0:
            padding = self.window_size - (exp.shape[-1] % self.window_size)

            padding_shape = [0] * len(exp.shape) * 2
            padding_shape[0] = padding

            if self.lut_build == "max":
                exp = torch.nn.functional.pad(exp, pad=tuple(padding_shape), value=127)
                mant = torch.nn.functional.pad(mant, pad=tuple(padding_shape), value=127)
            else:
                exp = torch.nn.functional.pad(exp, pad=tuple(padding_shape), value=-128)
                mant = torch.nn.functional.pad(mant, pad=tuple(padding_shape), value=-128)
            positive = torch.nn.functional.pad(positive, pad=tuple(padding_shape), value=False)
            negative = torch.nn.functional.pad(negative, pad=tuple(padding_shape), value=False)

        exp = exp.reshape(exp.shape[0], exp.shape[1] // self.window_size, self.window_size)
        mant = mant.reshape(mant.shape[0], mant.shape[1] // self.window_size, self.window_size)
        positive = positive.reshape(positive.shape[0], positive.shape[1] // self.window_size, self.window_size)
        negative = negative.reshape(negative.shape[0], negative.shape[1] // self.window_size, self.window_size)
        
        if self.lut_build == 'max':
            # calculate pos min and max windows
            pos_max_exp_window = torch.max(exp, dim = -1, keepdim=True)[0]
            pos_max_exp_window[pos_max_exp_window > self.max_exp] = self.max_exp
            pos_min_exp_window = pos_max_exp_window - (self.mant_dim - 1)
            pos_min_exp_window[pos_min_exp_window < self.min_exp] = self.min_exp

            # calculate neg min and max windows
            neg_max_exp_window = torch.max(exp, dim = -1, keepdim=True)[0]
            neg_max_exp_window[neg_max_exp_window > self.max_exp] = self.max_exp
            neg_min_exp_window = neg_max_exp_window - (self.mant_dim - 1)
            neg_min_exp_window[neg_min_exp_window < self.min_exp] = self.min_exp

            # compare to pos min and max values
            exp[positive] = torch.clamp(exp, max=pos_max_exp_window, min=pos_min_exp_window)[positive]
            mant[positive] = torch.where(exp <= pos_max_exp_window, mant, self.mant_dim - 1)[positive]
            mant[positive] = torch.where(exp >= pos_min_exp_window, mant, 0)[positive]
            
            # compare to pos min and max values
            exp[negative] = torch.clamp(exp, max=neg_max_exp_window, min=neg_min_exp_window)[negative]
            mant[negative] = torch.where(exp <= neg_max_exp_window, mant, self.mant_dim - 1)[negative]
            mant[negative] = torch.where(exp >= neg_min_exp_window, mant, 0)[negative]
        else:
            pos_min_exp_window = torch.min(exp, dim=-1, keepdim=True)[0]
            pos_min_exp_window[pos_min_exp_window < self.min_exp] = self.min_exp
            pos_max_exp_window = pos_min_exp_window + (self.mant_dim - 1)
            pos_max_exp_window[pos_max_exp_window > self.max_exp] = self.max_exp

            neg_min_exp_window = torch.min(exp, dim=-1, keepdim=True)[0]
            neg_min_exp_window[neg_min_exp_window < self.min_exp] = self.min_exp
            neg_max_exp_window = neg_min_exp_window + (self.mant_dim - 1)
            neg_max_exp_window[neg_max_exp_window > self.max_exp] = self.max_exp

            exp[positive] = torch.clamp(exp, max=pos_max_exp_window, min=pos_min_exp_window)[positive]
            mant[positive] = torch.where(exp >= pos_min_exp_window, mant, 0)[positive]
            mant[positive] = torch.where(exp <= pos_max_exp_window, mant, self.mant_dim - 1)[positive]

            exp[negative] = torch.clamp(exp, max=neg_max_exp_window, min=neg_min_exp_window)[negative]
            mant[negative] = torch.where(exp >= neg_min_exp_window, mant, 0)[negative]
            mant[negative] = torch.where(exp <= neg_max_exp_window, mant, self.mant_dim - 1)[negative]

        # Unpad and reshape tensors to original shape
        exp = exp.view(*exp.shape[:-2], -1)
        mant = mant.view(*mant.shape[:-2], -1)

        exp = exp[:, -input_shape[-1]:]
        mant = mant[:, -input_shape[-1]:]

        exp = exp.view(-1, input_shape[-1])
        mant = mant.view(-1, input_shape[-1])

        exp = exp[:inter_shape[0], :]
        mant = mant[:inter_shape[0], :]

        exp = exp.view(*input_shape)
        mant = mant.view(*input_shape)

        return exp, mant

    def nonlinear(self, x):
        # self.pos_lut = self.pos_lut.to(self.device)
        # self.neg_lut = self.neg_lut.to(self.device)
        # Split exponent and signed mantissa, bitshift mantissa to 4 bits (assumes leading 0).
        x = x.to(torch.bfloat16)

        zero_mask = (x == 0)
        positive = (x > 0)
        negative = (x < 0)

        mant, exp = torch.frexp(x)

        exp = exp.to(torch.int8)

        mant.mul_(16).round_().abs_()
        mant = mant.to(torch.int8)

        # Increment exponent where mantissa has overflow (i.e., mantissa is 16 / needs)
        exp = torch.where(mant == 16, exp + 1, exp)
        exp = torch.where(~zero_mask, exp - 1, exp)

        pos_exp_greater = positive & (exp > self.max_exp)
        pos_exp_less = positive & (exp < self.min_exp)
        neg_exp_greater = negative & (exp > self.max_exp)
        neg_exp_less = negative & (exp < self.min_exp)

        # Convert mantissa to unsigned 3 bit integer
        mant &= 0x7

        exp, mant = self.window_gelu_approx(exp, mant, positive, negative)

        mant = mant.to(torch.bfloat16)
        mant.div_(8).add_(1)
        mant = torch.where(negative, mant * -1, mant)
        mant[zero_mask] = 0

        exp[zero_mask] = 0
        exp = exp.to(torch.bfloat16)
        exp = torch.pow(2, exp)

        gelu = mant * exp
        del exp, mant

        gelu = torch.nn.functional.gelu(gelu)

        # Apply conditions for out of bounds values
        gelu[pos_exp_greater] = x[pos_exp_greater]
        gelu[pos_exp_less] = 0
        gelu[neg_exp_greater] = 0
        gelu[neg_exp_less] = 0
        gelu[zero_mask] = 0

        return gelu