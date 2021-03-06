import torch
from utils.loader import CalibrationData
from torch.utils.data import DataLoader


def layer_wise_search():
    data_set = CalibrationData(folder="H:/Dataset/Ship_Data/Ship_Four_CLS/calib", im_size=(32, 32))
    loader = DataLoader(data_set, num_workers=8, batch_size=100, shuffle=False)
    from utils import pipeline

    from model.mbv2 import MBV2Face
    from model.shipnet import ShipNet

    net = MBV2Face()
    net.load_state_dict(torch.load('./model/params.pth')['state_dict'])
    q = pipeline.Quantize(bit_width=8, loader=loader,
                          use_gpu=True, q_method='MSE')
    q.apply(net)

    net = ShipNet()
    net.load_state_dict(torch.load('./model/model_029.pth'))
    q = pipeline.Quantize(bit_width=8, loader=loader,
                          use_gpu=True, q_method='MSE')
    q.apply(net)


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def q_():
    # data_set = CalibrationData(folder="H:\Dataset\ImageNet\calib", im_size=(224, 224))
    # loader = DataLoader(data_set, num_workers=8, batch_size=100, shuffle=False)
    # from utils import pipeline
    # #
    # from model.mbv2 import MBV2Face
    # from model.backbone.mobilenetv2_q import InvertedResidual
    # from model.shipnet import ShipNet
    # # #
    # net = MBV2Face()
    #
    # data_set = CalibrationData(folder="H:\Dataset\Face\calib", im_size=(128, 128))
    # loader = DataLoader(data_set, num_workers=8, batch_size=100, shuffle=False)
    # net.load_state_dict(torch.load('./model/params.pth')['state_dict'])
    #
    # for m in net.modules():
    #     if isinstance(m, InvertedResidual):
    #         m.turn_on_add()
    #
    # q = pipeline.Quantize(bit_width=8, loader=loader,
    #                       use_gpu=True, q_method='Naive')
    #
    # q.apply(net)

    # data_set = CalibrationData(folder="H:/Dataset/Ship_Data/Ship_Four_CLS/calib", im_size=(32, 32))
    # loader = DataLoader(data_set, num_workers=8, batch_size=100, shuffle=False)
    # from utils import pipeline
    #
    # net = ShipNet()
    # net.load_state_dict(torch.load('./model/model_029.pth'))
    # q = pipeline.Quantize(bit_width=8, loader=loader,
    #                       use_gpu=True, q_method='MSE')
    #
    # q.apply(net)

    # root = "H:\Dataset\ImageNet\ILSVRC_2012\calib"
    # dst = "H:\Dataset\ImageNet\calib"
    # import os
    # import random
    # import shutil
    # dir_list = os.listdir(root)
    #
    # idx = 0
    # for item in dir_list:
    #     sub_path = os.path.join(root, item)
    #     sub_files = os.listdir(sub_path)
    #     random.shuffle(sub_files)
    #     cnt = 0
    #     for _item in sub_files:
    #         shutil.copy(os.path.join(sub_path, _item), os.path.join(dst, "{:>03d}.JPEG".format(idx)))
    #         idx += 1
    #         cnt += 1
    #         if cnt >= 1000:
    #             break


    # config_example = {"calibration_path": "/",
    #                   "loader_workers": 4,
    #                   "input_size": (32, 32),
    #                   }
    #
    # import json
    # with open("config/example.json", "w") as fp:
    #     pass

    # from torchvision.models.mobilenet import mobilenet_v2
    #

    # from model.mobilenet_v2_q import mobilenet_v2, InvertedResidual
    # net = mobilenet_v2(pretrained=True)
    #
    # data_set = CalibrationData(folder="H:\Dataset\ImageNet\calib", im_size=(224, 224), mean=[0.485, 0.456, 0.406],
    #                            std=[0.229, 0.224, 0.225])
    # loader = DataLoader(data_set, num_workers=8, batch_size=100, shuffle=False)
    # # net.load_state_dict(torch.load('./model/params.pth')['state_dict'])
    # net.load_state_dict(torch.load('model/pretrained/mobilenet_v2-b0353104.pth'))
    # for m in net.modules():
    #     if isinstance(m, InvertedResidual):
    #         m.turn_on_add()
    #
    # q = pipeline.Quantize(bit_width=8, loader=loader,
    #                       use_gpu=True, q_method='Naive')
    #
    # q.apply(net)

    # from model.mobilenet import MobileNet_v1
    # #
    # net = MobileNet_v1()
    # net.load_state_dict(torch.load('./model/mobilenet_v1_1.0_224.pth'))
    # q = pipeline.Quantize(bit_width=8, loader=loader,
    #                       use_gpu=True, q_method='Naive')
    # q.apply(net)


    # from utils import pipeline
    # #
    # from zoo.mobilenet.mobilenet_v2_q import mobilenet_v2
    # from zoo.mobilenet.mobilenet_v2_q import InvertedResidual
    #
    # net = mobilenet_v2()
    #
    # data_set = CalibrationData(folder="../../Dataset/QuanDataset/", im_size=(224,224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  #
    # loader = DataLoader(data_set, num_workers=8, batch_size=100, shuffle=False)
    # # net.load_state_dict(torch.load('./model/mobilenet_v2-b0353104.pth')['state_dict'])
    # net.load_state_dict(torch.load('./model/MobileNet_V2/mobilenet_v2-b0353104.pth'))
    #
    # for m in net.modules():
    #     if isinstance(m, InvertedResidual):
    #         m.turn_on_add()
    #
    # q = pipeline.Quantize(bit_width=8, loader=loader,
    #                       use_gpu=True, q_method='Naive_channel', report_path='./report')
    # # q_method can choose from ['Naive', 'MSE', 'Heuristic']
    #
    # q.apply(net)

    from utils import pipeline
    from zoo.mobilenet_v1.mobilenet_pytorch import MobileNet_v1

    net = MobileNet_v1()

    data_set = CalibrationData(folder="../../Dataset/QuanDataset/", im_size=(224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  #
    loader = DataLoader(data_set, num_workers=8, batch_size=100, shuffle=False)
    # net.load_state_dict(torch.load('./model/mobilenet_v2-b0353104.pth')['state_dict'])
    net.load_state_dict(torch.load('./model/MobileNet_V1/mobilenet_v1_1.0_224.pth'))

    # for m in net.modules():
    #     if isinstance(m, InvertedResidual):
    #         m.turn_on_add()

    q = pipeline.Quantize(bit_width=8, loader=loader,
                          use_gpu=True, q_method='MSE_channel', report_path='./report')
    # q_method can choose from ['Naive', 'MSE', 'Naive_channel', 'MSE_channel', 'Heuristic']

    q.apply(net)


if __name__ == '__main__':
    # from utils.loader import CalibrationData
    # from torch.utils.data import DataLoader
    # # data_set = CalibrationData(folder="H:/Dataset/Ship_Data/Ship_Four_CLS/calib", im_size=(32, 32))
    #
    q_()


