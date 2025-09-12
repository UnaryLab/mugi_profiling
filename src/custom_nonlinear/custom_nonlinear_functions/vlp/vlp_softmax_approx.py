import torch
from custom_nonlinear.custom_approx import CustomSoftmax

# Code functions with fp16 precision as input / output, and is not tested for other datatypes.
# Edit exp_dim to adjust the LUT size
# Edit max exp to adjust the maximum exponent of the LUT
class VLPSoftmax(CustomSoftmax):
    def __init__(self, layer, device, profile_path, profile_dims, profile=False):
        super(VLPSoftmax, self).__init__(layer, device, profile_path, profile_dims, profile)

    def set_params(self, max_min_exp, exp_dim, mant_dim, window_size, lut_build, config_path):
        self.config_path = config_path
        self.lut_build = lut_build
        self.exp_dim = exp_dim
        self.mant_dim = mant_dim
        self.mant_round = mant_dim * 2
        self.window_size = window_size

        if lut_build == 'max':
            self.min_exp = max_min_exp - (exp_dim - 1)
            self.max_exp = max_min_exp
        elif lut_build == 'min':
            self.min_exp = max_min_exp
            self.max_exp = max_min_exp + (exp_dim - 1)
        else:
            raise ValueError("lut_build must be 'max' or 'min'")

    def window_softmax_approx(self, exp, mant):
        input_shape = exp.shape

        exp = exp.view(-1, input_shape[-1])
        mant = mant.view(-1, input_shape[-1])

        attn_inter_shape = exp.shape
        # Pad tensors
        if exp.shape[0] % self.window_size != 0:
            padding = self.window_size - (exp.shape[0] % self.window_size)

            padding_shape = [0] * len(exp.shape) * 2
            padding_shape[-1] = padding

            if self.lut_build == "max":
                exp = torch.nn.functional.pad(exp, pad=tuple(padding_shape), value=-128)
                mant = torch.nn.functional.pad(mant, pad=tuple(padding_shape), value=-128)
            else:
                exp = torch.nn.functional.pad(exp, pad=tuple(padding_shape), value=127)
                mant = torch.nn.functional.pad(mant, pad=tuple(padding_shape), value=127)

        exp = exp.view(self.window_size, exp.shape[0] // self.window_size, exp.shape[1])
        mant = mant.view(self.window_size, mant.shape[0] // self.window_size, mant.shape[1])

        if self.lut_build == 'max':
            # process exp first
            max_exp_window = torch.max(exp, dim=0, keepdim=True)[0]
            max_exp_window[max_exp_window > self.max_exp] = self.max_exp
            min_exp_window = max_exp_window - (self.mant_dim - 1)
            min_exp_window[min_exp_window < self.min_exp] = self.min_exp

            mant = torch.where(exp <= max_exp_window, mant, self.mant_dim - 1)
            mant = torch.where(exp >= min_exp_window, mant, 0)
            exp = torch.where(exp <= max_exp_window, exp, max_exp_window)
            exp = torch.where(exp >= min_exp_window, exp, min_exp_window)
            del max_exp_window, min_exp_window
        else:
            # process exp first
            min_exp_window = torch.min(exp, dim=0, keepdim=True)[0]
            min_exp_window[min_exp_window < self.min_exp] = self.min_exp
            max_exp_window = min_exp_window + (self.mant_dim - 1)
            max_exp_window[max_exp_window > self.max_exp] = self.max_exp

            mant = torch.where(exp >= min_exp_window, mant, 0)
            mant = torch.where(exp <= max_exp_window, mant, self.mant_dim - 1)
            exp = torch.where(exp >= min_exp_window, exp, min_exp_window)
            exp = torch.where(exp <= max_exp_window, exp, max_exp_window)
            del max_exp_window, min_exp_window
        
        exp = exp.view(-1, input_shape[-1])
        mant = mant.view(-1, input_shape[-1])

        exp = exp[:attn_inter_shape[0], :]
        mant = mant[:attn_inter_shape[0], :]

        exp = exp.view(*input_shape)
        mant = mant.view(*input_shape)

        return exp, mant


    def nonlinear(self, attn_weights, dim=-1, dtype=torch.bfloat16):
        attn_weights = attn_weights.to(torch.bfloat16)
        # Find the max value and subtract it to prevent overflow (max value is 0)
        attn_weights_max = torch.max(attn_weights, dim = dim, keepdim = True)[0]
        attn_weights = attn_weights - attn_weights_max
        del attn_weights_max

        attn_mask = attn_weights != 0
        attn_inf = attn_weights <= -65500.0

        # Split exponent and signed mantissa, bitshift mantissa to 4 bits (assumes leading 0).
        mant, exp = torch.frexp(attn_weights)
        del attn_weights

        exp = exp.to(torch.int8)
        
        # Increment exponent where mantissa has overflow (i.e., mantissa is 16 / needs)
        mant.mul_(self.mant_round).round_().abs_()
        mant = mant.to(torch.int8)

        exp = torch.where(attn_mask, exp - 1, exp)
        exp = torch.where(mant == self.mant_round, exp + 1, exp)

        max_exp_mask = exp > self.max_exp

        # Convert mantissa to unsigned 3 bit integer
        mant &= self.mant_dim - 1

        # Remove inf values to increase window selection stability
        exp[attn_inf] = 0
        del attn_inf

        # Clamp exponent and adjust mantissa to fit within LUT range
        exp, mant = self.window_softmax_approx(exp, mant)

        # Postprocess 0 case and large exponent case
        exp = exp.to(torch.bfloat16)
        exp = torch.pow(2, exp)

        mant = mant.to(torch.bfloat16)
        mant.div_(self.mant_dim).add_(1).mul_(-1)

        #exponentials = torch.ldexp(mant, exp)
        attn_weights = mant * exp
        attn_weights.exp_()

        attn_weights[~attn_mask] = 1
        attn_weights[max_exp_mask] = 0

        # Calculate softmax output
        attn_weights_sum = torch.sum(attn_weights, dim = dim, keepdim = True)
        attn_weights.div_(attn_weights_sum)

        return attn_weights