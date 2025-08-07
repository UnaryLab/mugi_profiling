import torch
from custom_nonlinear.custom_approx import CustomGelu
import os

# Code functions with fp16 precision as input / output, and is not tested for other datatypes.
# Edit exp_dim to adjust the LUT size
# Edit max exp to adjust the maximum exponent of the LUT
class VLPGelu(CustomGelu):
    def __init__(self, exp_dim, max_pos_exp, window_size, layer, device, profile_path, profile_dims, blocks=None, keys=None, profile=False):
        super(VLPGelu, self).__init__(layer, device, profile_path, profile_dims, blocks, keys, profile)
        self.exp_dim = exp_dim
        self.max_pos_exp = max_pos_exp
        self.max_neg_exp = max_pos_exp
        self.window_size = window_size
        self.build_lut()

        self.mant_dim = 8

    def reset_lut(self, exp_dim, max_pos_exp, window_size):
        self.exp_dim = exp_dim
        self.max_pos_exp = max_pos_exp
        self.max_neg_exp = max_pos_exp
        self.window_size = window_size
        self.build_lut()

    def build_lut(self):
        self.build_pos_lut()
        self.build_neg_lut()

    def build_pos_lut(self):
        # Mantissa dimension of virtual LUT
        mant_dim = 8

        # shift min_exp by max_exp
        self.pos_min_exp = self.max_pos_exp - (self.exp_dim - 1)

        # generate exponent values of LUT
        exp_values = torch.arange(self.exp_dim).reshape(self.exp_dim, 1)
        exp_table = exp_values.expand(self.exp_dim, mant_dim)
        exp_table = exp_table + self.pos_min_exp

        # generate mantissa values of LUT
        mant_values = torch.arange(mant_dim)
        mant_table = ((mant_values.expand(self.exp_dim, mant_dim) / 8) + 1)

        # combine exponent and mantissa values
        exp_table = exp_table.to(torch.int32)
        mant_table = mant_table.to(torch.float32)
        lookup_table = torch.ldexp(mant_table, exp_table)

        # apply exp to create LUT
        self.pos_lut = torch.nn.functional.gelu(lookup_table).to(torch.bfloat16).to(self.device)

    def build_neg_lut(self):
        # Mantissa dimension of virtual LUT
        mant_dim = 8

        # shift min_exp by max_exp
        self.neg_min_exp = self.max_neg_exp - (self.exp_dim - 1)

        # generate exponent values of LUT
        exp_values = torch.arange(self.exp_dim).reshape(self.exp_dim, 1)
        exp_table = exp_values.expand(self.exp_dim, mant_dim)
        exp_table = exp_table + self.neg_min_exp

        # generate mantissa values of LUT
        mant_values = torch.arange(mant_dim)
        mant_table = ((mant_values.expand(self.exp_dim, mant_dim) / 8) + 1) * -1

        # combine exponent and mantissa values
        exp_table = exp_table.to(torch.int32)
        mant_table = mant_table.to(torch.float32)
        lookup_table = torch.ldexp(mant_table, exp_table)

        # apply exp to create LUT
        self.neg_lut = torch.nn.functional.gelu(lookup_table).to(torch.bfloat16).to(self.device)

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

            exp = torch.nn.functional.pad(exp, pad=tuple(padding_shape), value=127)
            mant = torch.nn.functional.pad(mant, pad=tuple(padding_shape), value=-128)
            positive = torch.nn.functional.pad(positive, pad=tuple(padding_shape), value=False)
            negative = torch.nn.functional.pad(negative, pad=tuple(padding_shape), value=False)

        exp = exp.reshape(inter_shape[0], inter_shape[1] // self.window_size, self.window_size)
        mant = mant.reshape(inter_shape[0], inter_shape[1] // self.window_size, self.window_size)
        positive = positive.reshape(inter_shape[0], inter_shape[1] // self.window_size, self.window_size)
        negative = negative.reshape(inter_shape[0], inter_shape[1] // self.window_size, self.window_size)

        # calculate pos min and max windows
        pos_max_exp_window = torch.max(exp, dim = -1, keepdim=True)[0]
        pos_max_exp_window[pos_max_exp_window > self.max_pos_exp] = self.max_pos_exp
        pos_min_exp_window = pos_max_exp_window - 7
        pos_min_exp_window[pos_min_exp_window < self.pos_min_exp] = self.pos_min_exp

        # calculate neg min and max windows
        neg_max_exp_window = torch.max(exp, dim = -1, keepdim=True)[0]
        neg_max_exp_window[neg_max_exp_window > self.max_neg_exp] = self.max_neg_exp
        neg_min_exp_window = neg_max_exp_window - 7
        neg_min_exp_window[neg_min_exp_window < self.neg_min_exp] = self.neg_min_exp

        # compare to pos min and max values
        exp[positive] = torch.clamp(exp, max=pos_max_exp_window, min=pos_min_exp_window)[positive]
        mant[positive] = torch.where(exp <= pos_max_exp_window, mant, self.mant_dim - 1)[positive]
        mant[positive] = torch.where(exp >= pos_min_exp_window, mant, 0)[positive]
        
        # compare to pos min and max values
        exp[negative] = torch.clamp(exp, max=neg_max_exp_window, min=neg_min_exp_window)[negative]
        mant[negative] = torch.where(exp <= neg_max_exp_window, mant, self.mant_dim - 1)[negative]
        mant[negative] = torch.where(exp >= neg_min_exp_window, mant, 0)[negative]

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

        mant.mul_(16)
        mant.round_()
        mant.abs_()
        mant = mant.to(torch.int8)

        # Increment exponent where mantissa has overflow (i.e., mantissa is 16 / needs)
        exp = torch.where(mant == 16, exp + 1, exp)
        exp = torch.where(~zero_mask, exp - 1, exp)

        pos_exp_greater = positive & (exp > self.max_pos_exp)
        pos_exp_less = positive & (exp < self.pos_min_exp)
        neg_exp_greater = negative & (exp > self.max_neg_exp)
        neg_exp_less = negative & (exp < self.neg_min_exp)

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