import torch
from utils.sim_tool import Simulation
from utils.pipeline import post_processing
# from zoo.mobilenet.mobilenet_v2_q import mobilenet_v2
from zoo.mobilenet_v1.mobilenet_pytorch import MobileNet_v1
from zoo.mobilenet.mobilenet_v2_q import InvertedResidual

# net = MobileNet_v1()
# net.load_state_dict(torch.load('./model/MobileNet_V1/mobilenet_v1_1.0_224.pth'))
net = torch.load('./result/pth/quantized_model.pth')
# for m in net.modules():
#     if isinstance(m, InvertedResidual):
#         m.turn_on_add()

post_processing(model=net, pth_path='result/pth/quantized.pth',
                info_path='report/info.csv', bit_width=8)
#
# sim = Simulation()
#
# sim.convert(net)
#
#
# # from model.shipnet import ShipNet
# # net = ShipNet()
# #
# # # net.load_state_dict(torch.load('./model/model_029.pth'))
# #
# # post_processing(model=net, pth_path='result/pth/quantized.pth',
# #                 info_path='report/info.csv', bit_width=8)
# #
# # sim = Simulation()
# #
# # sim.convert(net)

# from torchvision.models import mobilenet_v2
# from model.mobilenet_v2_q import mobilenet_v2, InvertedResidual
# net = mobilenet_v2(pretrained=True)
#
# for m in net.modules():
#     if isinstance(m, InvertedResidual):
#         m.turn_on_add()
#
# post_processing(model=net, pth_path='result/pth/quantized.pth',
#                 info_path='report/info.csv', bit_width=8)

sim = Simulation()

sim.convert(net)
