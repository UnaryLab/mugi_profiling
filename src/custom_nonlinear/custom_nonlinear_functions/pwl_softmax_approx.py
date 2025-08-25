import torch
from custom_nonlinear.custom_approx import CustomSoftmax
import os

# Code functions with fp16 precision as input / output, and is not tested for other datatypes.
# Edit segments to adjust the number of piecewise linear segments
# Edit segment_0 to set range (softmax range is set from segment_0 to 0 ex. -20 = [-20, 0])

class PWLSoftmax(CustomSoftmax):
    def __init__(self, segments, segment_0, layer, device, profile_path, profile_dims, blocks=None, keys=None, profile=False):
        super(PWLSoftmax, self).__init__(layer, device, profile_path, profile_dims, blocks, keys, profile)
        self.segments = torch.tensor(segments)
        self.segment_0 = torch.tensor(segment_0)
        self.segment_f = torch.tensor(0.0)
        self.device = device

        self.build_lut()

    def reset_lut(self, segments, segment_0):
        self.segments = torch.tensor(segments)
        self.segment_0 = torch.tensor(-segment_0)
        self.segment_f = torch.tensor(0.0)
        self.build_lut()

    def build_lut(self):
        self.step = torch.abs(self.segment_f - self.segment_0) / (self.segments)
        self.step = self.step.to(torch.float32)
    
    def nonlinear(self, attn_weights, dim=-1, dtype=torch.bfloat16):
        self.step = self.step.to(self.device)
        self.segment_0 = self.segment_0.to(self.device)
        self.segments = self.segments.to(self.device)

        attn_weights = attn_weights.to(torch.bfloat16)
        attn_weights_max = torch.max(attn_weights, dim = dim, keepdim = True)[0]
        attn_weights.sub_(attn_weights_max)
        del attn_weights_max

        x = attn_weights - self.segment_0
        x.div_(self.step)
        x.floor_()
        x.clamp_(min=0, max=self.segments.item())
        x_1 = x + 1

        x.mul_(self.step)
        x.add_(self.segment_0)

        x_1.mul_(self.step)
        x_1.add_(self.segment_0)

        m = torch.exp(x_1)
        del x_1
        b = torch.exp(x)

        m.sub_(b)
        m.div_(self.step)
        b.sub_(m * x)
        del x
        
        attn_mask = attn_weights < self.segment_0
        attn_weights.mul_(m)
        attn_weights.add_(b)
        #attn_weights = m * attn_weights + b
        del m, b

        attn_weights[attn_mask] = 0
        del attn_mask

        attn_weights_sum = torch.sum(attn_weights, dim = dim, keepdim = True)
        attn_weights.div_(attn_weights_sum)

        return attn_weights