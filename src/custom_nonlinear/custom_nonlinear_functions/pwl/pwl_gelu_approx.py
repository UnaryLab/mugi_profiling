import torch
from custom_nonlinear.custom_approx import CustomGelu
import os

# Code functions with fp16 precision as input / output, and is not tested for other datatypes.
# Edit segments to adjust the number of piecewise linear segments
# Edit segment_0 to set range (GeLU range is set on both sides ex. 4 = [-4, 4])

class PWLGelu(CustomGelu):
    def __init__(self, layer, device, profile_path, profile_dims, profile=False):
        super(PWLGelu, self).__init__(layer, device, profile_path, profile_dims, profile)

    def set_params(self, segments, segment_0, keys=None):
        self.keys = keys
        self.segments = torch.tensor(segments - 1)
        self.segment_0 = torch.tensor(-segment_0)
        self.segment_f = torch.tensor(segment_0)
        self.step = torch.abs(self.segment_f - self.segment_0) / (self.segments)
        self.step = self.step.to(torch.bfloat16)
    
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

        m = torch.nn.functional.gelu(x_1)
        del x_1
        b = torch.nn.functional.gelu(x_0)

        m.sub_(b).div_(self.step).sub_(m * x_0)
        del x_0

        gelu_output = m * x + b
        gelu_output[x < self.segment_0] = 0
        gelu_output[x > self.segment_f] = x[x > self.segment_f]

        return gelu_output