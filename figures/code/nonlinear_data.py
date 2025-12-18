import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
torch.set_printoptions(linewidth=120, edgeitems=32, sci_mode=False, threshold=(1600))


def window_softmax_approx(exp, mant):

    exp_dim = 12
    mant_dim = 8
    max_exp = 3
    min_exp = max_exp - (exp_dim - 1)
    exp_values = torch.arange(exp_dim).reshape(exp_dim, 1)
    exp_table = exp_values.expand(exp_dim, mant_dim)
    exp_table = exp_table + min_exp
    mant_values = torch.arange(mant_dim)
    mant_table = ((mant_values.expand(exp_dim, mant_dim) / 8) + 1) * -1
    lookup_table_fp = torch.ldexp(mant_table, exp_table).to(torch.float16)
    lookup_table_softmax = torch.exp(lookup_table_fp)

    window_size = 32
    reshape = int(math.ceil(exp.shape[-1] / window_size))

    if exp.shape[-1] % window_size != 0:
        padding = window_size - (exp.shape[-1] % window_size)
    else:
        padding  = 0

    # Pad tensors to create windows
    padded_exp = torch.nn.functional.pad(exp, (0, padding), "constant", -1000)
    padded_exp_shape = list(exp.shape[:-1]) + ([reshape, window_size])
    padded_exp = padded_exp.view(padded_exp_shape)
    padded_mant = torch.nn.functional.pad(mant, (0, padding), "constant", -1000)
    padded_mant_shape = list(mant.shape[:-1]) + ([reshape, window_size])
    padded_mant = padded_mant.view(padded_mant_shape)

    # calculate min and max windows
    max_exp_window = torch.max(padded_exp, dim = -1, keepdim=True)[0].expand_as(padded_exp)
    max_exp_window = torch.where(max_exp_window > max_exp, max_exp, max_exp_window)
    min_exp_window = max_exp_window - 7
    min_exp_window = torch.where(min_exp_window < min_exp, min_exp, min_exp_window)

    # compare to min and max values
    exp_window_max = torch.where(padded_exp <= max_exp_window, padded_exp, max_exp_window)
    exp_window = torch.where(exp_window_max >= min_exp_window, exp_window_max, torch.where(exp_window_max == -1000, exp_window_max, min_exp_window))
    
    mant_window_max = torch.where(padded_exp <= max_exp_window, padded_mant, 7)
    mant_window = torch.where(exp_window_max >= min_exp_window, mant_window_max, torch.where(exp_window_max == -1000, mant_window_max, 0))
    
    # reshape to original shape
    exp_windowed = exp_window.view(list(exp.shape[:-1]) +  [reshape * window_size])
    exp_out = exp_windowed[:,:,:,0:exp.shape[-1]] + abs(min_exp)
    mant_windowed = mant_window.view(list(mant.shape[:-1]) +  [reshape * window_size])
    mant_out = mant_windowed[:,:,:,0:mant.shape[-1]]
    return exp_out, mant_out

def exp_mugi():

    exp_dim = 12
    mant_dim = 8
    max_exp = 3
    min_exp = max_exp - (exp_dim - 1)
    exp_values = torch.arange(exp_dim).reshape(exp_dim, 1)
    exp_table = exp_values.expand(exp_dim, mant_dim)
    exp_table = exp_table + min_exp
    mant_values = torch.arange(mant_dim)
    mant_table = ((mant_values.expand(exp_dim, mant_dim) / 8) + 1) * -1
    lookup_table_fp = torch.ldexp(mant_table, exp_table).to(torch.float16)
    lookup_table_softmax = torch.exp(lookup_table_fp)

    x_1 = torch.linspace(-31, -7.5, 512).to(torch.bfloat16).view(1, 1, 1, -1)
    x_2 = torch.linspace(-7.5, -1.875, 512).to(torch.bfloat16).view(1, 1, 1, -1)
    x_3 = torch.linspace(-1.875, 0, 512).to(torch.bfloat16).view(1, 1, 1, -1)

    x = torch.cat((x_1, x_2, x_3), dim = -1)

    mant, exp = torch.frexp(x)
    mant_3bit = torch.round(mant * 16)

    exp = torch.where(mant_3bit == -16, exp + 1, exp)
    exp = torch.where(x == 0, exp, exp - 1)
    mant = torch.abs(mant_3bit.to(torch.int64)) & 0x7
    
    exp_preclamp = exp.to(torch.int64)
    
    exp_clamped, mant = window_softmax_approx(exp_preclamp, mant)


    exponentials = torch.where(x == 0, 1, lookup_table_softmax[exp_clamped, mant])
    exponentials = torch.where(exp_preclamp > max_exp, 0, exponentials)

    torch_exp = torch.exp(x)

    error = exponentials - torch_exp
    percent_error = (error / torch_exp) * 100
    percent_error = torch.where(error == torch_exp, 0, percent_error)
    percent_error = torch.where(percent_error > 100, 100, percent_error)
    percent_error = torch.where(x < -15, 100, percent_error)

    x = x.flatten()
    percent_error = percent_error.flatten()

    x = x.to(torch.float32)
    percent_error = percent_error.to(torch.float32)

    df = pd.DataFrame({'x': x, 'y': percent_error})
    df.to_csv('csv/nonlinear_error/exp_mugi.csv')

    # plt.plot(x, percent_error)
    # plt.grid()
    # plt.savefig('exp/mugi/exp_mugi_error_percent.png')
    # plt.close()

def create_lut(exp_dim, min_exp, neg = True, mant_dim = 8):

    exp_values = torch.arange(exp_dim).reshape(exp_dim, 1)
    exp_table = exp_values.expand(exp_dim, mant_dim)
    exp_table = exp_table + min_exp
    mant_values = torch.arange(mant_dim)
    mant_table = (mant_values.expand(exp_dim, mant_dim) / 8) + 1
    if(neg == True):
        mant_table = mant_table * -1
        #mant_table = torch.flip(mant_table, dims = [1])

    lookup_table = torch.ldexp(mant_table, exp_table).to(torch.bfloat16)
    lookup_table_silu = (lookup_table * torch.sigmoid(lookup_table))
    return lookup_table_silu

def create_lut_gelu(exp_dim, min_exp, neg = True, mant_dim = 8):

    exp_values = torch.arange(exp_dim).reshape(exp_dim, 1)
    exp_table = exp_values.expand(exp_dim, mant_dim)
    exp_table = exp_table + min_exp
    mant_values = torch.arange(mant_dim)
    mant_table = (mant_values.expand(exp_dim, mant_dim) / 8) + 1
    if(neg == True):
        mant_table = mant_table * -1
        #mant_table = torch.flip(mant_table, dims = [1])

    lookup_table = torch.ldexp(mant_table, exp_table).to(torch.bfloat16)
    lookup_table_gelu = torch.nn.functional.gelu(lookup_table)
    return lookup_table_gelu




def window_gelu_approx(exp, mant):
        
    exp_dim = 11
    max_exp_pos = 1
    max_exp_neg = 1
    max_exp = max_exp_pos
    min_exp = max_exp - (exp_dim - 1)
    min_exp_pos = max_exp_pos - (exp_dim - 1)
    min_exp_neg = max_exp_neg - (exp_dim - 1)
    lut_gelu_pos = create_lut_gelu(exp_dim, min_exp_pos, neg = False)
    lut_gelu_neg = create_lut_gelu(exp_dim, min_exp_neg, neg = True)
    window_size = 256

    reshape = int(math.ceil(exp.shape[-1] / window_size))

    if exp.shape[-1] % window_size != 0:
        padding = window_size - (exp.shape[-1] % window_size)
    else:
        padding  = 0

    # Pad tensors to create windows
    padded_exp = torch.nn.functional.pad(exp, (0, padding), "constant", -1000)
    padded_exp_shape = list(exp.shape[:-1]) + ([reshape, window_size])
    padded_exp = padded_exp.view(padded_exp_shape)
    padded_mant = torch.nn.functional.pad(mant, (0, padding), "constant", -1000)
    padded_mant_shape = list(mant.shape[:-1]) + ([reshape, window_size])
    padded_mant = padded_mant.view(padded_mant_shape)

    # calculate min and max windows
    max_exp_window = torch.max(padded_exp, dim = -1, keepdim=True)[0].expand_as(padded_exp)
    max_exp_window = torch.where(max_exp_window > max_exp, max_exp, max_exp_window)
    min_exp_window = max_exp_window - 7
    min_exp_window = torch.where(min_exp_window < min_exp, min_exp, min_exp_window)

    # compare to min and max values
    exp_window_max = torch.where(padded_exp <= max_exp_window, padded_exp, max_exp_window)
    exp_window = torch.where(exp_window_max >= min_exp_window, exp_window_max, torch.where(exp_window_max == -1000, exp_window_max, min_exp_window))
    
    mant_window_max = torch.where(padded_exp <= max_exp_window, padded_mant, min_exp_window)
    mant_window = torch.where(exp_window_max >= min_exp_window, mant_window_max, torch.where(exp_window_max == -1000, mant_window_max, 0))
    
    # reshape to original shape
    exp_windowed = exp_window.view(list(exp.shape[:-1]) +  [reshape * window_size])
    exp_out = exp_windowed[:,:,0:exp.shape[-1]] + abs(min_exp)
    mant_windowed = mant_window.view(list(mant.shape[:-1]) +  [reshape * window_size])
    mant_out = mant_windowed[:,:,0:mant.shape[-1]]
    return exp_out, mant_out

def gelu_approx_comp(input, unclamped_exp, exp, mant):

    exp_dim = 11
    max_exp_pos = 1
    max_exp_neg = 1
    max_exp = max_exp_pos
    min_exp = max_exp - (exp_dim - 1)
    min_exp_pos = max_exp_pos - (exp_dim - 1)
    min_exp_neg = max_exp_neg - (exp_dim - 1)
    lut_gelu_pos = create_lut_gelu(exp_dim, min_exp_pos, neg = False)
    lut_gelu_neg = create_lut_gelu(exp_dim, min_exp_neg, neg = True)
    window_size = 256

    gelu = torch.where(input > 0,
                        torch.where(unclamped_exp > exp, input, lut_gelu_pos[exp, mant]),
                        lut_gelu_neg[exp, mant])
    gelu = torch.where(input < 0,
                        torch.where(unclamped_exp < exp, 0, lut_gelu_neg[exp, mant]),
                        gelu)
    gelu = torch.where(input == 0, 0, gelu)
    return gelu

def gelu_mugi():

    exp_dim = 11
    max_exp_pos = 1
    max_exp_neg = 1
    max_exp = max_exp_pos
    min_exp = max_exp - (exp_dim - 1)
    min_exp_pos = max_exp_pos - (exp_dim - 1)
    min_exp_neg = max_exp_neg - (exp_dim - 1)
    lut_gelu_pos = create_lut_gelu(exp_dim, min_exp_pos, neg = False)
    lut_gelu_neg = create_lut_gelu(exp_dim, min_exp_neg, neg = True)
    window_size = 256

    x = torch.linspace(-5, 5, 512).to(torch.bfloat16).view(1, 1, 1, -1)
    mant, exp = torch.frexp(x)

    exp = exp - 1
    mant *= 16

    mant = torch.where(exp > 0, mant, torch.round(mant))
    exp = torch.where(mant >= 16, exp + 1,
                        torch.where(mant <= -16, exp + 1, exp))

    mant = torch.abs(mant.to(torch.int64)) & 0x7

    exp_clamped, mant = window_gelu_approx(exp, mant)

    exp_window = exp_clamped.to(torch.int64)
    gelu = gelu_approx_comp(x, exp + abs(min_exp), exp_window, mant).flatten()
    x = x.flatten()

    torch_gelu = torch.nn.functional.gelu(x).flatten()

    error = (gelu - torch_gelu)
    percent_error = (error / torch_gelu) * 100
    percent_error = torch.where(error == torch_gelu, 0, percent_error)
    percent_error = torch.where(x > .01, percent_error, torch.where(x < -.01, percent_error, 0))
    percent_error = torch.where(percent_error > 100, 100, percent_error)

    x = x.to(torch.float32)
    percent_error = percent_error.to(torch.float32)

    df = pd.DataFrame({'x': x, 'y': percent_error})

    df.to_csv('csv/nonlinear_error/gelu_mugi.csv')

    # plt.plot(x, percent_error)
    # plt.grid()
    # plt.savefig('gelu/mugi/gelu_mugi_error_percent.png')
    # plt.close()



# --------------------------------------

def window_silu_approx(exp, mant):
        
    exp_dim = 11
    max_exp_pos = 1
    max_exp_neg = 1
    max_exp = max_exp_pos
    min_exp = max_exp - (exp_dim - 1)
    min_exp_pos = max_exp_pos - (exp_dim - 1)
    min_exp_neg = max_exp_neg - (exp_dim - 1)
    lut_silu_pos = create_lut(exp_dim, min_exp_pos, neg = False)
    lut_silu_neg = create_lut(exp_dim, min_exp_neg, neg = True)
    window_size = 256

    reshape = int(math.ceil(exp.shape[-1] / window_size))

    if exp.shape[-1] % window_size != 0:
        padding = window_size - (exp.shape[-1] % window_size)
    else:
        padding  = 0

    # Pad tensors to create windows
    padded_exp = torch.nn.functional.pad(exp, (0, padding), "constant", -1000)
    padded_exp_shape = list(exp.shape[:-1]) + ([reshape, window_size])
    padded_exp = padded_exp.view(padded_exp_shape)
    padded_mant = torch.nn.functional.pad(mant, (0, padding), "constant", -1000)
    padded_mant_shape = list(mant.shape[:-1]) + ([reshape, window_size])
    padded_mant = padded_mant.view(padded_mant_shape)

    # calculate min and max windows
    max_exp_window = torch.max(padded_exp, dim = -1, keepdim=True)[0].expand_as(padded_exp)
    max_exp_window = torch.where(max_exp_window > max_exp, max_exp, max_exp_window)
    min_exp_window = max_exp_window - 7
    min_exp_window = torch.where(min_exp_window < min_exp, min_exp, min_exp_window)

    # compare to min and max values
    exp_window_max = torch.where(padded_exp <= max_exp_window, padded_exp, max_exp_window)
    exp_window = torch.where(exp_window_max >= min_exp_window, exp_window_max, torch.where(exp_window_max == -1000, exp_window_max, min_exp_window))
    
    mant_window_max = torch.where(padded_exp <= max_exp_window, padded_mant, min_exp_window)
    mant_window = torch.where(exp_window_max >= min_exp_window, mant_window_max, torch.where(exp_window_max == -1000, mant_window_max, 0))
    
    # reshape to original shape
    exp_windowed = exp_window.view(list(exp.shape[:-1]) +  [reshape * window_size])
    exp_out = exp_windowed[:,:,0:exp.shape[-1]] + abs(min_exp)
    mant_windowed = mant_window.view(list(mant.shape[:-1]) +  [reshape * window_size])
    mant_out = mant_windowed[:,:,0:mant.shape[-1]]
    return exp_out, mant_out

def silu_approx_comp(input, unclamped_exp, exp, mant):

    exp_dim = 11
    max_exp_pos = 1
    max_exp_neg = 1
    max_exp = max_exp_pos
    min_exp = max_exp - (exp_dim - 1)
    min_exp_pos = max_exp_pos - (exp_dim - 1)
    min_exp_neg = max_exp_neg - (exp_dim - 1)
    lut_silu_pos = create_lut(exp_dim, min_exp_pos, neg = False)
    lut_silu_neg = create_lut(exp_dim, min_exp_neg, neg = True)
    window_size = 256

    silu = torch.where(input > 0,
                        torch.where(unclamped_exp > exp, input, lut_silu_pos[exp, mant]),
                        lut_silu_neg[exp, mant])
    silu = torch.where(input < 0,
                        torch.where(unclamped_exp < exp, 0, lut_silu_neg[exp, mant]),
                        silu)
    silu = torch.where(input == 0, 0, silu)
    return silu

def silu_mugi():

    exp_dim = 11
    max_exp_pos = 1
    max_exp_neg = 1
    max_exp = max_exp_pos
    min_exp = max_exp - (exp_dim - 1)
    min_exp_pos = max_exp_pos - (exp_dim - 1)
    min_exp_neg = max_exp_neg - (exp_dim - 1)
    lut_silu_pos = create_lut(exp_dim, min_exp_pos, neg = False)
    lut_silu_neg = create_lut(exp_dim, min_exp_neg, neg = True)
    window_size = 256

    x = torch.linspace(-5, 5, 512).to(torch.bfloat16).view(1, 1, 1, -1)
    mant, exp = torch.frexp(x)

    exp = exp - 1
    mant *= 16

    mant = torch.where(exp > 0, mant, torch.round(mant))
    exp = torch.where(mant >= 16, exp + 1,
                        torch.where(mant <= -16, exp + 1, exp))

    mant = torch.abs(mant.to(torch.int64)) & 0x7

    exp_clamped, mant = window_silu_approx(exp, mant)

    exp_window = exp_clamped.to(torch.int64)
    silu = silu_approx_comp(x, exp + abs(min_exp), exp_window, mant).flatten()
    x = x.flatten()

    torch_silu = torch.nn.functional.silu(x).flatten()

    error = (silu - torch_silu)
    percent_error = (error / torch_silu) * 100
    percent_error = torch.where(error == torch_silu, 0, percent_error)
    percent_error = torch.where(x > .01, percent_error, torch.where(x < -.01, percent_error, 0))
    percent_error = torch.where(percent_error > 100, 100, percent_error)

    x = x.to(torch.float32)
    percent_error = percent_error.to(torch.float32)

    df = pd.DataFrame({'x': x, 'y': percent_error})
    df.to_csv('csv/nonlinear_error/silu_mugi.csv')

    # plt.plot(x, percent_error)
    # plt.grid()
    # plt.savefig('silu/mugi/silu_mugi_error_percent.png')
    # plt.close()

def exp_taylor():
    degrees = 9
    a = -4.5

    prev_exp = torch.tensor(1).to(torch.float32)
    x_neg = torch.tensor(0).to(torch.float32)
    exp = 0
    while True:
        for i in range(degrees):
            intermediate = x_neg - a
            intermediate = intermediate ** i
            intermediate = intermediate / math.factorial(i)
            exp += intermediate
        exp *= (torch.e ** (a))
        exp = exp.to(torch.bfloat16)
        if exp <= 0.001 or prev_exp < exp or torch.isinf(exp):
            break
        else:
            prev_exp = exp.clone()
            x_neg -= 0.25
    
    x_1 = torch.linspace(-31, -7.5, 512).to(torch.bfloat16)
    x_2 = torch.linspace(-7.5, -1.875, 512).to(torch.bfloat16)
    x_3 = torch.linspace(-1.875, 0, 512).to(torch.bfloat16)
    
    x = torch.cat((x_1, x_2, x_3))
    exp = torch.zeros(x.size())
    for i in range(degrees + 1):
        intermediate = x - a
        intermediate = intermediate ** i
        intermediate = intermediate / math.factorial(i)
        exp += intermediate
    exp *= (torch.e ** (a))
    exp = torch.where(x < x_neg, 0, exp)

    torch_exp = torch.exp(x)

    error = (exp - torch_exp)
    percent_error = (error / torch_exp) * 100
    percent_error = torch.where(error == torch_exp, 0, percent_error)

    x = x.to(torch.float32)
    percent_error = percent_error.to(torch.float32)

    df = pd.DataFrame({'x': x, 'y': percent_error})
    df.to_csv('csv/nonlinear_error/exp_taylor.csv')

    # plt.plot(x, percent_error)
    # plt.grid()
    # plt.savefig('exp/taylor/exp_taylor_error_percent.png')
    # plt.close()

def ymxb(y0, y1, x0, x1):
    m = (y1 - y0) / (x1 - x0)
    b = y0 - m * x0
    return m, b

def exp_pwl():
    segments = 22
    segment_0 = -15
    segment_f = 0

    x_segments = torch.linspace(segment_0, segment_f, segments + 1).to(torch.bfloat16)
    y_segments = torch.exp(x_segments).to(torch.bfloat16)

    mb = [ymxb(y_segments[i], y_segments[i+1], x_segments[i], x_segments[i+1]) for i in range(0, segments)]
    m = [mb[i][0].item() for i in range(0, segments)]
    b = [mb[i][1].item() for i in range(0, segments)]

    x_1 = torch.linspace(-31, -7.5, 512).to(torch.bfloat16)
    x_2 = torch.linspace(-7.5, -1.875, 512).to(torch.bfloat16)
    x_3 = torch.linspace(-1.875, 0, 512).to(torch.bfloat16)
    x = torch.cat((x_1, x_2, x_3))

    y = torch.tensor([0.0] * len(x)).to(torch.bfloat16)
    
    for i in range(segments):
        y = torch.where(x >= x_segments[i], m[i] * x + b[i], y)
    y = torch.where(x < segment_0, 0, y)
    y = torch.where(x > x_segments[-1], m[-1] * x + b[-1], y)

    torch_exp = torch.exp(x)

    error = (y - torch_exp)
    percent_error = (error / torch_exp) * 100
    percent_error = torch.where(error == torch_exp, 0, percent_error)

    average_error = error.mean()
    median_error = torch.median(error)
    average_percent_error = percent_error.mean()
    median_percent_error = torch.median(percent_error)
    print('exp_pwl')
    print('average error:', average_error.item())
    print('median error:', median_error.item())
    print('average percent error:', average_percent_error.item())
    print('median percent error:', median_percent_error.item(), end='\n\n')

    x = x.to(torch.float32)
    percent_error = percent_error.to(torch.float32)

    df = pd.DataFrame({'x': x, 'y': percent_error})
    df.to_csv('csv/nonlinear_error/exp_pwl.csv')

    # plt.plot(x, percent_error)
    # plt.grid()
    # plt.savefig('exp/pwl/exp_pwl_error_percent.png')
    # plt.close()

def exp_torch():
    x_1 = torch.linspace(-15, -7.5, 512).to(torch.bfloat16)
    x_2 = torch.linspace(-7.5, -1.875, 512).to(torch.bfloat16)
    x_3 = torch.linspace(-1.875, -1, 512).to(torch.bfloat16)

    x = torch.cat((x_1, x_2))
    exp = torch.exp(x)
    x = x.to(torch.float16)
    exp = exp.to(torch.float16)
    df = pd.DataFrame({'x': x, 'y': exp})
    df.to_csv('csv/nonlinear_error/exp_torch.csv')

    # plt.plot(x, exp)
    # plt.grid()
    # plt.savefig('exp/torch/exp_torch.png')
    # plt.close()

def silu_torch():
    x = torch.linspace(-5, 5, 512).to(torch.bfloat16)
    y = torch.nn.functional.silu(x)
    x = x.to(torch.float16)
    y = y.to(torch.float16)

    df = pd.DataFrame({'x': x, 'y': y})
    df.to_csv('csv/nonlinear_error/silu_torch.csv')

    # plt.plot(x, y)
    # plt.grid()
    # plt.savefig('silu/torch/silu_torch.png')
    # plt.close()

def gelu_torch():
    x = torch.linspace(-5, 5, 512).to(torch.bfloat16)
    y = torch.nn.functional.gelu(x)
    x = x.to(torch.float16)
    y = y.to(torch.float16)

    df = pd.DataFrame({'x': x, 'y': y})
    df.to_csv('csv/nonlinear_error/gelu_torch.csv')

    # plt.plot(x, y)
    # plt.grid()
    # plt.savefig('gelu/torch/gelu_torch.png')
    # plt.close()

def silu_pwl():
    segments = 22
    segment_0 = -5
    segment_f = 5

    x_segments = torch.linspace(segment_0, segment_f, segments + 1).to(torch.bfloat16)
    y_segments = torch.nn.functional.silu(x_segments).to(torch.bfloat16)
    y_segments[0] = 0

    mb = [ymxb(y_segments[i], y_segments[i+1], x_segments[i], x_segments[i+1]) for i in range(0, segments)]
    m = [mb[i][0].item() for i in range(0, segments)]
    b = [mb[i][1].item() for i in range(0, segments)]

    x = torch.linspace(segment_0, segment_f, 512).to(torch.bfloat16)
    y = torch.tensor([0.0] * len(x)).to(torch.bfloat16)
    
    for i in range(len(x)):
        for j in range(segments):
            if x[i] >= x_segments[j]:
                y[i] = m[j] * x[i] + b[j]
    
    torch_silu = torch.nn.functional.silu(x)

    error = (y - torch_silu)
    percent_error = (error / torch_silu) * 100
    percent_error = torch.where(error == torch_silu, 0, percent_error)

    x = x.to(torch.float16)
    percent_error = percent_error.to(torch.float16)
    df = pd.DataFrame({'x': x, 'y': percent_error})
    df.to_csv('csv/nonlinear_error/silu_pwl.csv')

    # plt.plot(x, percent_error)
    # plt.grid()
    # plt.savefig('silu/pwl/silu_pwl_error_percent.png')
    # plt.close()

def gelu_pwl():
    segments = 22
    segment_0 = -5
    segment_f = 5

    x_segments = torch.linspace(segment_0, segment_f, segments + 1).to(torch.bfloat16)
    y_segments = torch.nn.functional.gelu(x_segments).to(torch.bfloat16)
    y_segments[0] = 0

    mb = [ymxb(y_segments[i], y_segments[i+1], x_segments[i], x_segments[i+1]) for i in range(0, segments)]
    m = [mb[i][0].item() for i in range(0, segments)]
    b = [mb[i][1].item() for i in range(0, segments)]

    x = torch.linspace(segment_0, segment_f, 512).to(torch.bfloat16)
    y = torch.tensor([0.0] * len(x)).to(torch.bfloat16)
    
    for i in range(len(x)):
        for j in range(segments):
            if x[i] >= x_segments[j]:
                y[i] = m[j] * x[i] + b[j]
    
    torch_gelu = torch.nn.functional.gelu(x)

    error = (y - torch_gelu)
    percent_error = (error / torch_gelu) * 100
    percent_error = torch.where(error == torch_gelu, 0, percent_error)

    x = x.to(torch.float16)
    percent_error = percent_error.to(torch.float16)
    df = pd.DataFrame({'x': x, 'y': percent_error})

    df.to_csv('csv/nonlinear_error/gelu_pwl.csv')

    # plt.plot(x, percent_error)
    # plt.grid()
    # plt.savefig('gelu/pwl/gelu_pwl_error_percent.png')
    # plt.close()

def silu_pwl_dep():
    x = torch.linspace(-5, 5, 512).to(torch.bfloat16)
    y = torch.where(x > 0, x, 0)
    df = pd.DataFrame({'x': x, 'y': y})
    df.to_csv('csv/nonlinear_error/silu_pwl.csv')

    # plt.plot(x, y)
    # plt.grid()
    # plt.savefig('silu/pwl/silu_pwl.png')
    # plt.close()

def silu_mobilenet_pwl():
    x = torch.linspace(-5, 5, 512).to(torch.bfloat16)
    y = x * (torch.nn.functional.relu6(x + 3) / 6)
    x = x.to(torch.float16)
    y = y.to(torch.float16)
    

    torch_silu = torch.nn.functional.silu(x)

    error = (y - torch_silu)
    percent_error = (error / torch_silu) * 100
    percent_error = torch.where(error == torch_silu, 0, percent_error)

    df = pd.DataFrame({'x': x, 'y': percent_error})
    df.to_csv('csv/nonlinear_error/silu_mobilenet_pwl.csv')

    # plt.plot(x, percent_error)
    # plt.grid()
    # plt.savefig('silu/pwl/silu_mobilenet_pwl_error_percent.png')
    # plt.close()

def error(base_path, approx_path, error_path):
    base = pd.read_csv(base_path)
    approx = pd.read_csv(approx_path)
    
    # Ensure the data is read correctly
    if 'y' not in base.columns or 'y' not in approx.columns:
        raise ValueError("CSV files must contain 'y' column")
    
    error = (base['y'] - approx['y']).astype(np.float64)
    error = error.abs()

    epsilon = 1e-10  # Small value to avoid division by zero
    percent_error = ((error / (base['y'] + epsilon)).abs() * 100).astype(np.float64)
    percent_error = np.where(error == base['y'], 0, percent_error)

    max_error = max(base['y'].abs().max(), approx['y'].abs().max())

    error_dif = ((error / max_error) * 100).astype(np.float64)

    avg_percent_error = percent_error.mean()
    median_percent_error = np.median(percent_error)
    print(approx_path)
    print('avg percent error: {:.2e}'.format(avg_percent_error))  # Print in scientific notation
    print('median percent error: {:.2e}'.format(median_percent_error), end='\n\n')  # Print in scientific notation

    df = pd.DataFrame({'x': base['x'], 'y': error})
    df.to_csv(error_path + '.csv')
    df = pd.DataFrame({'x': base['x'], 'y': percent_error})
    df.to_csv(error_path + '_percent.csv')
    df = pd.DataFrame({'x': base['x'], 'y': error_dif})
    df.to_csv(error_path + '_dif.csv')

    plt.plot(base['x'].to_numpy(), error.to_numpy())
    #plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.grid()
    plt.savefig(error_path + '.png')
    plt.close()

    plt.plot(base['x'].to_numpy(), percent_error)
    #plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.grid()
    plt.savefig(error_path + '_percent.png')
    plt.close()

if not os.path.exists('csv/nonlinear_error'):
    os.makedirs('csv/nonlinear_error')

exp_taylor()
exp_pwl()
exp_mugi()

gelu_pwl()
gelu_mugi()

silu_pwl()
silu_mobilenet_pwl()
silu_mugi()

# error('exp/torch/exp_torch.csv', 'exp/taylor/exp_taylor.csv', 'exp/taylor/exp_taylor_error')
# error('exp/torch/exp_torch.csv', 'exp/mugi/exp_mugi.csv', 'exp/mugi/exp_mugi_error')
# error('exp/torch/exp_torch.csv', 'exp/pwl/exp_pwl.csv', 'exp/pwl/exp_pwl_error')
# error('silu/torch/silu_torch.csv', 'silu/pwl/silu_pwl.csv', 'silu/pwl/silu_pwl_error')
# error('silu/torch/silu_torch.csv', 'silu/pwl_mobilenet/silu_mobilenet_pwl.csv', 'silu/pwl_mobilenet/silu_mobilenet_pwl_error')
# error('silu/torch/silu_torch.csv', 'silu/mugi/silu_mugi.csv', 'silu/mugi/silu_mugi_error')