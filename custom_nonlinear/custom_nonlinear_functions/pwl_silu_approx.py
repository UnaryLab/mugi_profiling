import torch
from custom_nonlinear.custom_approx import CustomSilu
import os

# Code functions with fp16 precision as input / output, and is not tested for other datatypes.
# Edit segments to adjust the number of piecewise linear segments
# Edit segment_0 to set range (SiLU range is set on both sides ex. 4 = [-4, 4])

class PWLSilu(CustomSilu):
    def __init__(self, segments, segment_0, layer, device, profile_path, profile_dims, blocks=None, keys=None, profile=False):
        super(PWLSilu, self).__init__(layer, device, profile_path, profile_dims, blocks, keys, profile)
        self.segments = torch.tensor(segments)
        self.segment_0 = torch.tensor(-segment_0)
        self.segment_f = torch.tensor(segment_0)
        self.device = device

        self.build_lut()

    def reset_lut(self, segments, segment_0):
        self.segments = torch.tensor(segments - 1)
        self.segment_0 = torch.tensor(-segment_0)
        self.segment_f = torch.tensor(segment_0)

        self.build_lut()

    def build_lut(self):
        self.step = torch.abs(self.segment_f - self.segment_0) / (self.segments)
        self.step = self.step.to(torch.float32)
    
    def nonlinear(self, x):
        self.step = self.step.to(self.device)
        self.segment_0 = self.segment_0.to(self.device)
        self.segments = self.segments.to(self.device)

        x = x.to(torch.bfloat16)

        x_0 = x - self.segment_0
        x_0.div_(self.step)
        x_0.floor_()
        x_0.clamp_(min=0, max=self.segments.item())
        x_1 = x_0 + 1

        x_0.mul_(self.step)
        x_0.add_(self.segment_0)

        x_1.mul_(self.step)
        x_1.add_(self.segment_0)

        m = torch.nn.functional.silu(x_1)
        del x_1
        b = torch.nn.functional.silu(x_0)

        m.sub_(b)
        m.div_(self.step)
        b.sub_(m * x_0)
        del x_0

        silu_output = m * x + b
        silu_output[x < self.segment_0] = 0
        silu_output[x > self.segment_f] = x[x > self.segment_f]

        return silu_output