import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from .layer import QConv2d, search_replace_convolution2d, QAvgPooling, QAddition
from .absorb_bn import search_absorb_bn
from .optimizer import mse_minimize_quantize, naive_scaling_quantize, mse_channel_quantize, naive_channel_quantize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Quantize(object):
    """
    The Quantize class provides main quantization pipeline.
    Currently, layer-wise quantization is implemented.

    There are several properties defined here:
    (1) bit_width: the bit-width of the feature map
    (2) per_layer: the quantization scaling factors are imposed on layer-wise.
        In the future, we will include channel-wise quantization.
    (3) q_method: quantization algorithm used in quantization pipeline.

    """
    def __init__(self, bit_width, loader=None,
                 per_layer=True, q_method='MSE',
                 use_gpu=False, report_path=None,
                 pth_save_path=None):
        assert bit_width in [8, 4], "bit width should be 8 or 4."
        assert isinstance(per_layer, bool), "per_layer should be a boolean."
        assert q_method in ['MSE', 'Naive', 'Heuristic', 'MSE_channel', 'Naive_channel']

        self.bit_width = bit_width
        self.per_layer = per_layer
        self.q_method = q_method

        self.max_value = 2**(bit_width-1) - 1
        self.min_value = -2**(bit_width-1)

        self.loader = loader
        self.use_gpu = use_gpu

        if os.path.isdir(report_path):
            self.report_path = report_path
        else:
            print("There is no report_path")

        # self.report_path = report_path
        self.pth_path = pth_save_path

    @torch.no_grad()
    def apply(self, model):
        assert isinstance(model, torch.nn.Module)

        model.eval()
        print("================= Search all BN layers and remove. ============")
        search_absorb_bn(model)

        print("=========== Search all convolution layers and replace. ========")
        search_replace_convolution2d(model, self.bit_width)
        mark_layer(model, 'root')

        print("=========== Perform layer-wise statistics. ====================")
        for m in model.modules():
            if isinstance(m, QConv2d):
                QConv2d.quantized = False

        if self.use_gpu:
            model = model.cuda()

        cnt = 0
        res_report = []
        total_time = 0.0
        for m in model.modules():
            if (isinstance(m, QConv2d) and not m.quantized) or (
                    isinstance(m, QAddition) and not m.quantized):
                start_time = time.time()
                print("==> Start quantizing layer: {} ".format(m.name))

                cached_input = []
                cached_output = []

                cached_input_lhs = []
                cached_input_rhs = []

                if isinstance(m, QConv2d):
                    layer_type = 'CONV'

                    def save_hook_conv(_, _input, _output): #  hook(model, input, output)
                        cached_input.append(_input[0].detach().cpu())  # detach, no_grad
                        cached_output.append(_output.detach().cpu())
                    handle = m.register_forward_hook(save_hook_conv)
                else:
                    layer_type = 'ADD'

                    def save_hook_add(_, _input, _output):
                        cached_input_lhs.append(_input[0].detach().cpu())
                        cached_input_rhs.append(_input[1].detach().cpu())
                        cached_output.append(_output.detach().cpu())
                    handle = m.register_forward_hook(save_hook_add)

                for im in self.loader:
                    if self.use_gpu:
                        im = im.cuda()
                    model(im)
                handle.remove()  # save_hook must be removed.

                if isinstance(m, QConv2d):
                    _x = torch.cat(cached_input, dim=0)
                    _w = m.weight.detach().cpu()
                    _o = torch.cat(cached_output, dim=0)
                else:
                    _x = torch.cat(cached_input_lhs, dim=0)
                    _w = torch.cat(cached_input_rhs, dim=0)
                    _o = torch.cat(cached_output, dim=0)
                s1, s2, s3 = self.quantize(_x, _w, _o, m)

                # print("S1: {:>2.2f}".format(s1), " S2: {:>2.2f}".format(s2))
                m.x_quantizer.s.data = s1
                m.w_quantizer.s.data = s2
                m.quantized = True

                res_report.append([cnt, m.name, np.array(s1), np.array(s2), np.array(s3), layer_type])
                cnt += 1

                end_time = time.time()
                print("Time: {:>4.2f} s".format(end_time - start_time))
                total_time += (end_time - start_time)
        csv_writer_mv1(res_report)
        # csv_writer_mv2(res_report)
        torch.save(model, './result/pth/quantized_model.pth')
        torch.save(model.state_dict(), './result/pth/quantized.pth')

        print("Total Time: {:>4.2f}".format(total_time))
        print("\n==> Check quantization report and correct parameters.")

    def quantize(self, x, w, o, layer):
        if self.q_method == 'MSE_channel':
            return self.channel_quantize(x, w, o, layer)
        elif self.q_method == 'Naive_channel':
            return self.naive_channel(x, w, o, layer)
        elif self.q_method == 'MSE':
            return self.mse_quantize(x, w, o, layer)
        elif self.q_method == 'Naive':
            return self.naive_quantize(x, w, o, layer)
        else:
            raise RuntimeError("Unknown Quantization Method.")

    def mse_quantize(self, x, w, o, layer):
        return mse_minimize_quantize(x, w, o, layer, self.max_value, self.min_value,
                                     device=['cpu', 'gpu'][self.use_gpu])

    def naive_quantize(self, x, w, _, __):
        return naive_scaling_quantize(x, w, None, None, self.max_value, self.min_value)

    def naive_channel(self, x, w, o, layer):
        return naive_channel_quantize(x, w, o, layer, self.max_value, self.min_value,
                                      device=['cpu', 'gpu'][self.use_gpu])

    def channel_quantize(self, x, w, o, layer):
        return mse_channel_quantize(x, w, o, layer, self.max_value, self.min_value,
                                    device=['cpu', 'gpu'][self.use_gpu])


def mark_layer(model, name):
    cnt = 0
    for child_name, child in model.named_children():
        if isinstance(child, QConv2d) or isinstance(child, QAvgPooling) or \
                isinstance(child, QAddition):
            child.name = name + '.' + str(cnt)
            cnt += 1
        else:
            mark_layer(child, name + '.' + child_name)


def csv_writer_mv1(res):
    assert isinstance(res, list)
    info_len = len(res)
    extended_res = []
    for _idx in range(info_len):
        tmp = res[_idx]
        src_id, layer_name, s1, s2, s3_, layer_type = tmp  # unpack layer info.
        s1, s2 = s1.tolist(), s2.tolist(),
        # s3_ = res[(_idx + 1) % info_len][2]
        s3 = []
        if (_idx != 26) & (_idx != 27):
            for _i in range(len(s1)):
                s3.append(res[(_idx+1) % info_len][2][_i])
        elif _idx == 26:
            for _i in range(len(s1)):
                s3.append(res[(_idx+1) % info_len][2][0])
        elif _idx == 27:
            s3 = torch.tensor(np.ones([1001])*50).tolist()
            # s3 = torch.tensor(np.ones([1001])*50).tolist()
        # s3 = s3_.tolist()0
        extended_res.append([src_id, layer_name, s1, s2, s3,
                             layer_type, (_idx+1) % info_len])
    extended_res[info_len-1][4] = torch.tensor(np.ones([1001])*50).tolist()
    df = pd.DataFrame(extended_res, columns=["SRC", "Name", "S1", "S2", "S3", "TYPE", "DST"])
    df.to_csv('./report/info.csv', index=False)

def csv_writer_mv2(res):
    assert isinstance(res, list)
    info_len = len(res)
    extended_res = []
    for _idx in range(info_len):
        tmp = res[_idx]
        src_id, layer_name, s1, s2, s3_, layer_type = tmp  # unpack layer info.
        s1, s2 = s1.tolist(), s2.tolist(),
        # s3_ = res[(_idx + 1) % info_len][2]
        s3 = []
        if (_idx != 8) & (_idx != 15) & (_idx != 19) & (_idx != 26) & (_idx != 30) & (_idx != 34) & (_idx != 41) & (_idx != 45) & (_idx != 52) & (_idx != 56) & (_idx != 62):
            for _i in range(len(s1)):
                s3.append(res[(_idx+1) % info_len][2][0])
        elif (_idx != 8) or (_idx != 15) or (_idx != 19) or (_idx != 26) or (_idx != 30) or (_idx != 34) or (_idx != 41) or (_idx != 45) or (_idx != 52) or (_idx != 56):
            for _i in range(len(s1)):
                s3.append(res[(_idx+1) % info_len][3][0])
        elif _idx == 62:
            s3 = torch.tensor(np.ones([1000])*1).tolist()
            # s3 = torch.tensor(np.ones([1001])*50).tolist()
        # s3 = s3_.tolist()0
        extended_res.append([src_id, layer_name, s1, s2, s3,
                             layer_type, (_idx+1) % info_len])
    extended_res[info_len-1][4] = torch.tensor(np.ones([1000])*50).tolist()
    df = pd.DataFrame(extended_res, columns=["SRC", "Name", "S1", "S2", "S3", "TYPE", "DST"])
    df.to_csv('./report/info.csv', index=False)

def post_processing(model, pth_path, info_path, bit_width):
    import numpy as np
    import scipy.io as sio
    from .base import mul_shift_calculator, pure_quantize, scaling_align_calculator, pure_quantize_weight_channel, pure_quantize_weight

    assert isinstance(model, nn.Module)

    df = pd.read_csv(info_path)
    q_info = dict()
    for idx, row in df.iterrows():
        s1_, s2_, s3_ = row['S1'].strip('[]').split(','), row['S2'].strip('[]').split(','), row['S3'].strip('[]').split(',')
        s1 = [(float(s1_[_i])) for _i in range(len(s1_))]
        s2 = [(float(s2_[_i])) for _i in range(len(s2_))]
        s3 = [(float(s3_[_i])) for _i in range(len(s3_))]
        q_info[row['Name']] = [s1, s2, s3]

    model.eval()
    # search_absorb_bn(model)
    # search_replace_convolution2d(model, bit_width)
    # mark_layer(model, 'root')
    #
    # model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    print('==> Params loaded.')
    mat_file = []
    for m in model.modules():
        if isinstance(m, QConv2d):
            scaling = q_info[m.name]
            mat_params = dict()
            mul, shift = mul_shift_calculator(scaling[0], scaling[1], scaling[2])
            q_weight = pure_quantize_weight_channel(m.weight, scaling[1], bit_width)

            mat_params['Name'] = m.name
            q_weight = q_weight.permute(2, 3, 1, 0)

            if m.groups > 1:
                q_weight = q_weight.squeeze(dim=2)

            mat_params['Weight'] = q_weight.detach().cpu().numpy()

            if m.bias is not None:
                q_bias = pure_quantize_weight(m.bias, scaling[0], scaling[1], bit_width * 2)
                q_bias = q_bias.detach().cpu().numpy()
                bias_abs = torch.Tensor(m.bias.size())
                for _i in range(m.bias.size()[0]):
                    bias_abs[_i] = torch.abs(m.bias[_i] * scaling[0][_i] * scaling[1][_i])
                    if bias_abs[_i] > 2**(2*bit_width - 1) - 1:
                    # if bias_abs[_i] > 2**(4*bit_width - 1) - 1:

                    # if bias_abs.max() > 2 ** (2 * bit_width - 1) - 1:
                        print("==> Layer: {} Bias overflow.{}".format(m.name, _i))
                mat_params['Bias'] = q_bias

            mat_params['Stride'] = np.asarray(m.stride)
            mat_params['Groups'] = float(m.groups)
            mat_params['Padding'] = np.asarray(m.padding)
            mat_params['Mul'] = mul
            mat_params['Shift'] = shift

            m.mul.data = torch.tensor(mul)
            m.shift.data = torch.tensor(shift)

            mat_file.append(mat_params)

        if isinstance(m, QAddition):
            scaling = q_info[m.name]
            mat_params = dict()

            mul_lhs, mul_rhs, s_lr = scaling_align_calculator(torch.Tensor(scaling[2])/torch.Tensor(scaling[0]), torch.Tensor(scaling[2])/torch.Tensor(scaling[1]))

            mat_params['Name'] = m.name
            mat_params['Mul_L'] = mul_lhs
            mat_params['Mul_R'] = mul_rhs
            mat_params['Shift_L'] = s_lr
            mat_params['Shift_R'] = s_lr

            m.mul_lhs.data = torch.tensor(mul_lhs)
            m.mul_rhs.data = torch.tensor(mul_rhs)

            m.shift_lhs.data = torch.tensor(s_lr)
            m.shift_rhs.data = torch.tensor(s_lr)

            mat_file.append(mat_params)

        if isinstance(m, nn.AvgPool2d) or isinstance(m, nn.MaxPool2d) \
                or isinstance(m, QAvgPooling):
            mat_params = dict()
            mat_params['Name'] = "Pooling"
            mat_params['Stride'] = np.asarray(m.stride)
            mat_params['Kernel'] = np.asarray(m.kernel_size)
            mat_params['Type'] = m.__repr__()[:9]

            mat_file.append(mat_params)
    sio.savemat('./result/mat/quantized.mat', {'Net': mat_file})
    torch.save(model.state_dict(), './result/pth/true_quantized.pth')
    torch.save(model, './result/pth/true_quantized_model.pth')
