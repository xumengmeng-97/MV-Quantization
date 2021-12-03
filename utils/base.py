import torch
import numpy as np


def mul_shift_calculator(s1, s2, s3):
    assert isinstance(s1, list) and isinstance(s2, list) \
           and isinstance(s3, list)
    for __i in range(len(s3)):
        if s3[__i] == 0:
            s3[__i] = 10e-7  # eps for safe division.
    mul_ = []
    shift_ = []
    for _i in range(len(s3)):
        re_scale = 1 / (s1[_i] * s2[_i] / s3[_i])
        shift_list = list(range(1, 23))
        m_s_list = []
        for shift in shift_list:
            mul = np.round((2 ** shift) * re_scale)
            err = np.abs(mul / (2 ** shift) - re_scale)
            m_s_list.append([mul, shift, err])

        mul_range_safe = False
        final_list = []
        for item in m_s_list:
            if 100 <= item[0] <= 512:
                mul_range_safe = True
                final_list.append(item)
        if mul_range_safe is False:
            print("Multiplier is out of numeric range.")
            err_res = np.asarray(m_s_list).astype(np.float32)
        else:
            err_res = np.asarray(final_list).astype(np.float32)
        idx = err_res[:, 2].argmin()
        mul, shift = float(err_res[idx][0]), float(err_res[idx][1])
        mul_.append(mul)
        shift_.append(shift)
    return mul_, shift_

def scaling_align_calculator(s3_1, s3_2):
    mul_l_ = []
    mul_r_ = []
    shift_ = []
    for _i in range(s3_1.size()[0]):
        shift_range = list(range(1, 23))
        m_s_list = []
        for shift in shift_range:
            mul_l = np.round((2 ** shift) * s3_1[_i])
            mul_r = np.round((2 ** shift) * s3_2[_i])
            err = np.abs(mul_l / (2 ** shift) - s3_1[_i]) + np.abs(mul_r / (2 ** shift) - s3_2[_i])
            m_s_list.append([mul_l, mul_r, shift, err])
        mul_range_safe = False
        final_list = []
        for item in m_s_list:
            if 900 <= item[0] <= 4096 and 900 <= item[1] <= 4096:
                mul_range_safe = True
                final_list.append(item)

        if mul_range_safe is False:
            print("Multiplier is out of numeric range.")
            err_res = np.asarray(m_s_list).astype(np.float32)
        else:
            err_res = np.asarray(final_list).astype(np.float32)

        idx = err_res[:, 3].argmin()
        mul_l, mul_r, shift = float(err_res[idx][0]), float(err_res[idx][1]), float(err_res[idx][2])
        mul_l_.append(mul_l)
        mul_r_.append(mul_r)
        shift_.append(shift)
    return mul_l_, mul_r_, shift_


def pure_quantize(t, s, bit_width):
    max_v = 2 ** (bit_width - 1) - 1
    min_v = -2 ** (bit_width - 1)
    return torch.round(torch.clamp(t * s, min_v, max_v))

def pure_quantize_weight(t, s1, s2, bit_width):
    max_v = 2 ** (bit_width - 1) - 1
    min_v = -2 ** (bit_width - 1)
    t_ = torch.Tensor(t.size())
    t_ = t_.cuda()
    for _i in range(t.size()[0]):
        t_[_i] = t[_i] * s1[_i] * s2[_i]
    return torch.round(torch.clamp(t_, min_v, max_v))

def pure_quantize_weight_channel(t, s, bit_width):
    max_v = 2 ** (bit_width - 1) - 1
    min_v = -2 ** (bit_width - 1)
    t_ = torch.Tensor(t.size())
    for _i in range(t.size()[0]):
        t_[_i, :, :, :] = t[_i, :, :, :] * s[_i]
    return torch.round(torch.clamp(t_, min_v, max_v))
