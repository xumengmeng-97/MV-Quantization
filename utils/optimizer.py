import torch
import numpy as np
import torch.nn as nn
from .layer import QConv2d, QAddition


@torch.no_grad()
def mse_minimize_quantize(x, w, o, layer, max_v, min_v, device='gpu'):
    """
    The algorithm utilized to determine optimal scaling factors for x and w is proposed in paper:

    M. Zhao, K. Ning, S. Yu, L. Liu and N. Wu, "Quantizing Oriented Object Detection Network via
    Outlier-Aware Quantization and IoU Approximation," in IEEE Signal Processing Letters,
    doi: 10.1109/LSP.2020.3031490.

    Some details are slightly different from the original paper, where we search not only the scaling
    factors for weight but also for input features. Hence, the search space is two-dimensional, indicating
    a more time-consuming procedure as well as a more accurate approximation.

    :param x: input feature map of a given convolution layer, should be a tensor.
    :param w: weight of a given convolution layer, should be a tensor.
    :param o: output feature map of a given convolution layer, should be a tensor.
    :param layer: the convolution layer that need to be quantized.
    :param max_v: the max numeric value of a certain bit-width, e.g. 127 of 8-bit signed integer.
    :param min_v: the same as max_v.
    :param device: which device is used.
    :return: scaling factors for input (x) and weight (w).
    """
    assert isinstance(layer, QConv2d) or isinstance(layer, QAddition)
    assert isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor) and isinstance(o, torch.Tensor)

    beta = 0.85
    alpha = 5
    tolerance = 1.5
    sample_points = 60

    criterion = nn.MSELoss()

    "Estimate the optimal scaling factor range."
    std_w = torch.sqrt(w.var()).item()
    std_x = torch.sqrt(x.var()).item()
    mean_w = torch.mean(w).item()
    mean_x = torch.mean(x).item()

    sx0, sw0 = naive_scaling_quantize(x, w, None, None, max_v, min_v)
    sx0 = sx0.item()
    sw0 = sw0.item()

    sx_min, sw_min = sx0 * beta, sw0 * beta
    # sx_min, sw_min = sx0, sw0

    sx_max = max((sx0 * tolerance, max_v / (alpha * std_x)))
    sw_max = max((sw0 * tolerance, max_v / (alpha * std_w)))
    # sx_max = max((sx0 * tolerance, max_v / (mean_x + alpha * std_x)))
    # sw_max = max((sw0 * tolerance, max_v / (mean_w + alpha * std_w)))

    " Uniformly sampling in the scaling factor range. "
    s_x_list = [(sx_min + (_item + 1) * (sx_max - sx_min) / sample_points) for _item in range(sample_points)]
    s_w_list = [(sw_min + (_item + 1) * (sw_max - sw_min) / sample_points) for _item in range(sample_points)]

    if device != 'cpu':
        x = x.cuda()
        o = o.cuda()
        if isinstance(layer, QAddition):
            w = w.cuda()

    err_array = []

    layer.x_quantizer.s.data = torch.tensor(sx0)
    layer.w_quantizer.s.data = torch.tensor(sw0)

    layer.use_quantization_simulation()

    if isinstance(layer, QConv2d):
        err0 = criterion(layer(x), o).detach().item()
    else:
        err0 = criterion(layer(x, w), o).detach().item()

    print("Initial MSE error: {:>4.4f}".format(float(err0)))
    print("Searching ...")

    with torch.no_grad():
        for _s_x in s_x_list:
            for _s_w in s_w_list:
                layer.x_quantizer.s.data = torch.tensor(_s_x)
                layer.w_quantizer.s.data = torch.tensor(_s_w)

                if isinstance(layer, QConv2d):
                    err = criterion(layer(x), o)
                else:
                    err = criterion(layer(x, w), o)

                err_ = err.detach().item()
                err_array.append([_s_x, _s_w, float(err_)])

    err_array = np.asarray(err_array).astype(np.float32)
    idx = err_array[:, 2].argmin()

    if float(err_array[idx, 2]) > float(err0):
        err_final = err0
        s1, s2 = sx0, sw0
    else:
        err_final = err_array[idx, 2]
        s1, s2 = err_array[idx, 0], err_array[idx, 1]

    print("Searched optimal MSE error: {:>4.4f}".format(err_final))
    return torch.tensor(float(s1)), torch.tensor(float(s2))

def mse_channel_quantize(x, w, o, layer, max_v, min_v, device='gpu'):
    assert isinstance(layer, QConv2d) or isinstance(layer, QAddition)
    assert isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor) and isinstance(o, torch.Tensor)

    scale_begin = 0.85
    x_scale_end = x.mean() + x.var()
    scale_end = 1.5
    sample_point = 40
    mul_scale_w = np.linspace(scale_begin, scale_end, sample_point)
    mul_scale_x = np.linspace(scale_begin, scale_end, sample_point)
    w_list = []
    o_list = []
    std_w = []
    criterion = nn.MSELoss()
    for _i in range(w.size()[0]):
        w_list.append(w[_i, :, :, :])
        std_w.append(torch.sqrt(w[_i, :, :, :].var()).item())
    for _i in range(o.size()[1]):
        o_list.append(o[:, _i, :, :])

    "Estimate the optimal scaling factor range."
    # std_x = torch.sqrt(x.var()).item()
    if x.size() == w.size() == o.size():
        sx0, sw0, so0 = naive_scaling_channel_quantize_add(x, w, None, None, max_v, min_v, o)
    else:
        sx0, sw0, so0 = naive_scaling_channel_quantize(x, w, None, None, max_v, min_v, o)

    if device != 'cpu':
        x = x.cuda()
        o = o.cuda()
        if isinstance(layer, QAddition):
            w = w.cuda()
    err0_scale = np.zeros([sample_point, sample_point])
    for i in range(len(mul_scale_x)):
        for j in range(len(mul_scale_w)):
            layer.x_quantizer.s.data = sx0 * mul_scale_x[i]
            layer.w_quantizer.s.data = sw0 * mul_scale_w[j]
            layer.use_quantization_simulation() # self.q_inference = True, self.q_inference_with_output = False
            if isinstance(layer, QConv2d):
                err0 = criterion(layer(x), o).detach().item()
            else:
                err0 = criterion(layer(x, w), o).detach().item()
            # err0_scale.append(err0)
            err0_scale[i][j] = err0

    [[min_index_x, min_index_w]] = np.argwhere(err0_scale == np.min(err0_scale))
    # min_index = err0_scale.index(min(err0_scale))
    layer.x_quantizer.s.data = sx0 * mul_scale_x[min_index_x]
    layer.w_quantizer.s.data = sw0 * mul_scale_w[min_index_w]
    if isinstance(layer, QConv2d):
        err = criterion(layer(x), o).detach().item()
    else:
        err = criterion(layer(x, w), o).detach().item()
    print("Naive error in : {:>4.10f}".format(float(err)))
    return sx0 * mul_scale_x[min_index_x], sw0 * mul_scale_w[min_index_w], so0

def naive_channel_quantize(x, w, o, layer, max_v, min_v, device='gpu'):
    assert isinstance(layer, QConv2d) or isinstance(layer, QAddition)
    assert isinstance(x, torch.Tensor) and isinstance(w, torch.Tensor) and isinstance(o, torch.Tensor)

    w_list = []
    o_list = []
    std_w = []
    criterion = nn.MSELoss()
    for _i in range(w.size()[0]):
        w_list.append(w[_i, :, :, :])
        std_w.append(torch.sqrt(w[_i, :, :, :].var()).item())
    for _i in range(o.size()[1]):
        o_list.append(o[:, _i, :, :])

    "Estimate the optimal scaling factor range."
    std_x = torch.sqrt(x.var()).item()
    if x.size() == w.size() == o.size():
        sx0, sw0, so0 = naive_scaling_channel_quantize_add(x, w, None, None, max_v, min_v, o)
    else:
        sx0, sw0, so0 = naive_scaling_channel_quantize(x, w, None, None, max_v, min_v, o)

    if device != 'cpu':
        x = x.cuda()
        o = o.cuda()
        if isinstance(layer, QAddition):
            w = w.cuda()

    layer.x_quantizer.s.data = sx0
    layer.w_quantizer.s.data = sw0

    layer.use_quantization_simulation() # self.q_inference = True, self.q_inference_with_output = False
    if isinstance(layer, QConv2d):
        err0 = criterion(layer(x), o).detach().item()
    else:
        err0 = criterion(layer(x, w), o).detach().item()

    print("Naive error in : {:>4.10f}".format(float(err0)))
    return sx0, sw0, so0

def naive_scaling_quantize(x, w, _, __, max_v, min_v) -> \
        (torch.Tensor, torch.Tensor):
    assert isinstance(x, torch.Tensor)
    assert isinstance(w, torch.Tensor)

    def find_scale(_x):
        max_in, min_in = _x.max(), _x.min()
        if torch.abs(max_in) > torch.abs(min_in):
            s1 = max_v / max_in
        else:
            s1 = min_v / min_in
        return s1

    return find_scale(x), find_scale(w)

def naive_scaling_channel_quantize(x, w, _, __, max_v, min_v, o) -> \
        (torch.Tensor, torch.Tensor):
    assert isinstance(x, torch.Tensor)
    assert isinstance(w, torch.Tensor)

    def find_scale_x(_x, _w):
        _x_channel = _w.size()[0]
        s_x = []
        for _i in range(_x_channel):
            max_x_, min_x_ = _x.max(), _x.min()
            if (max_x_ == 0) & (min_x_ == 0):
                s1 = 1
            elif torch.abs(max_x_) > torch.abs(min_x_):
                s1 = max_v / max_x_
            else:
                s1 = min_v / min_x_
            s1 = torch.clamp(torch.Tensor([s1]), 0, 250)
            s_x.append(s1.item())
        return torch.tensor(s_x)
        # if _w.size()[1] != 1:
        #     _x_channel = _w.size()[0]
        #     s_x = []
        #     for _i in range(_x_channel):
        #         max_x_, min_x_ = _x.max(), _x.min()
        #         if (max_x_ == 0) & (min_x_ == 0):
        #             s1 = 30
        #         elif torch.abs(max_x_) > torch.abs(min_x_):
        #             s1 = max_v / max_x_
        #         else:
        #             s1 = min_v / min_x_
        #         s1 = torch.clamp(torch.Tensor([s1]), 0, 250)
        #         s_x.append(s1.item())
        #     return torch.tensor(s_x)
        # elif _w.size()[1] == 1:
        #     _x_channel = _w.size()[0]
        #     s_x = []
        #     for _i in range(_x_channel):
        #         max_x_, min_x_ = _x[:, _i, :, :].max(), _x[:, _i, :, :].min()
        #         if (max_x_ == 0) & (min_x_ == 0):
        #             s1 = 30
        #         elif torch.abs(max_x_) > torch.abs(min_x_):
        #             s1 = max_v / max_x_
        #         else:
        #             s1 = min_v / min_x_
        #         s1 = torch.clamp(torch.Tensor([s1]), 0, 250)
        #         s_x.append(s1.item())
        #     return torch.tensor(s_x)
    def find_scale_w(_w):
        if _w.size()[1] != 1:
            _w_channel = _w.size()[0]
            s_w = []
            for _i in range(_w_channel):
                max_x_, min_x_ = _w[_i, :, :, :].max(), _w[_i, :, :, :].min()
                # max_x_, min_x_ = _w.max(), _w.min()
                if (max_x_ == 0) & (min_x_ == 0):
                    s1 = 30
                elif torch.abs(max_x_) > torch.abs(min_x_):
                    s1 = max_v / max_x_
                else:
                    s1 = min_v / min_x_
                s1 = torch.clamp(torch.Tensor([s1]), 0, 250)
                s_w.append(s1.item())
            return torch.tensor(s_w)
        elif _w.size()[1] == 1: # dw
            _w_channel = _w.size()[0]
            s_w = []
            for _i in range(_w_channel):
                max_x_, min_x_ = _w[_i, :, :, :].max(), _w[_i, :, :, :].min()
                # max_x_, min_x_ = _w.max(), _w.min()
                if (max_x_ == 0) & (min_x_ == 0):
                    s1 = 30
                elif torch.abs(max_x_) > torch.abs(min_x_):
                    s1 = max_v / max_x_
                else:
                    s1 = min_v / min_x_
                s1 = torch.clamp(torch.Tensor([s1]), 0, 250)
                s_w.append(s1.item())
            return torch.tensor(s_w)
    def find_scale_o(_o):
        _x_channel = _o.size()[1]
        s_x = []
        for _i in range(_x_channel):
            # max_x_, min_x_ = _o[:, _i, :, :].max(), _o[:, _i, :, :].min()
            max_x_, min_x_ = _o.max(), _o.min()
            if (max_x_ == 0) & (min_x_ == 0):
                s1 = 30
            # elif torch.abs(max_x_) > torch.abs(min_x_):
            else:
                s1 = abs(max_v / max_x_)
            # else:
            #     s1 = abs(min_v / min_x_)
            s1 = torch.clamp(torch.Tensor([s1]), 0, 250)
            s_x.append(s1.item())
        return torch.tensor(s_x)
    s_x, s_w, s_o = find_scale_x(x, w), find_scale_w(w), find_scale_o(o)
    def xwo_clamp(s_x, s_w, s_o):
        _c = s_x.size()[0]
        for _i in range(_c):
            if torch.abs(s_x[_i] * s_w[_i]) > 2660:
                s_w[_i] = 2660 / s_x[_i]
        return s_x, s_w, s_o
    return xwo_clamp(s_x, s_w, s_o)

def naive_scaling_channel_quantize_add(x, w, _, __, max_v, min_v, o) -> \
        (torch.Tensor, torch.Tensor):
    assert isinstance(x, torch.Tensor)
    assert isinstance(w, torch.Tensor)

    def find_scale_x(_x):
        _x_channel = _x.size()[1]
        s_x = []
        for _i in range(_x_channel):
            max_x_, min_x_ = _x.max(), _x.min()
            if (max_x_ == 0) & (min_x_ == 0):
                s1 = 1
            elif torch.abs(max_x_) > torch.abs(min_x_):
                s1 = max_v / max_x_
            else:
                s1 = min_v / min_x_
            s1 = torch.clamp(torch.Tensor([s1]), 0, 250)
            s_x.append(s1.item())
        return torch.tensor(s_x)
        # if _w.size()[1] != 1:
        #     _x_channel = _w.size()[0]
        #     s_x = []
        #     for _i in range(_x_channel):
        #         max_x_, min_x_ = _x.max(), _x.min()
        #         if (max_x_ == 0) & (min_x_ == 0):
        #             s1 = 1
        #         elif torch.abs(max_x_) > torch.abs(min_x_):
        #             s1 = max_v / max_x_
        #         else:
        #             s1 = min_v / min_x_
        #         s1 = torch.clamp(torch.Tensor([s1]), 0, 250)
        #         s_x.append(s1.item())
        #     return torch.tensor(s_x)
        # elif _w.size()[1] == 1:
        #     _x_channel = _w.size()[0]
        #     s_x = []
        #     for _i in range(_x_channel):
        #         max_x_, min_x_ = _x[:, _i, :, :].max(), _x[:, _i, :, :].min()
        #         if (max_x_ == 0) & (min_x_ == 0):
        #             s1 = 1
        #         elif torch.abs(max_x_) > torch.abs(min_x_):
        #             s1 = max_v / max_x_
        #         else:
        #             s1 = min_v / min_x_
        #         s1 = torch.clamp(torch.Tensor([s1]), 0, 250)
        #         s_x.append(s1.item())
        #     return torch.tensor(s_x)
    def find_scale_o(_o):
        _x_channel = _o.size()[1]
        s_x = []
        for _i in range(_x_channel):
            # max_x_, min_x_ = _o[:, _i, :, :].max(), _o[:, _i, :, :].min()
            max_x_, min_x_ = _o.max(), _o.min()
            if (max_x_ == 0) & (min_x_ == 0):
                s1 = 1
            # elif torch.abs(max_x_) > torch.abs(min_x_):
            else:
                s1 = abs(max_v / max_x_)
            # else:
            #     s1 = abs(min_v / min_x_)
            s1 = torch.clamp(torch.Tensor([s1]), 0, 250)
            s_x.append(s1.item())
        return torch.tensor(s_x)
    s_x, s_w, s_o = find_scale_x(x), find_scale_x(w), find_scale_o(o)
    def xwo_clamp(s_x, s_w, s_o):
        _c = s_x.size()[0]
        for _i in range(_c):
            if torch.abs(s_x[_i] * s_w[_i]) > 3000:
                s_w[_i] = 3000 / s_x[_i]
        return s_x, s_w, s_o
    return xwo_clamp(s_x, s_w, s_o)