import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
# from model.shipnet import ShipNet
from utils.absorb_bn import search_absorb_bn
from utils.layer import search_replace_convolution2d, QConv2d, QAvgPooling
from utils.pipeline import mark_layer

from zoo.mobilenet.mobilenet_v2_q import mobilenet_v2, InvertedResidual
from zoo.mobilenet_v1.mobilenet_pytorch import MobileNet_v1
from utils.layer import QAddition, search_turn_off_relu6
from utils.helper import AverageMeter, accuracy
from torch.utils.data import dataloader
from torchvision.datasets import ImageFolder


def check_correctness():
    model = mobilenet_v2()

    # model.load_state_dict(torch.load('../model/model_029.pth'))
    model.eval()
    search_absorb_bn(model)
    search_replace_convolution2d(model, 8)
    mark_layer(model, 'root')

    model.load_state_dict(torch.load('./result/pth/true_quantized.pth'))
    model.eval()

    test_process = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                       ])

    for m in model.modules():
        if isinstance(m, QConv2d):
            m.use_full_quantization()
        if isinstance(m, QAvgPooling):
            m.q_inference = True

    import scipy.io as sio
    a = sio.loadmat('model/sample/test.mat')['img']
    im = Image.fromarray(a)
    im = test_process(im)[None]
    from utils.layer import QuantizeLayer

    q_input = QuantizeLayer(8, s=torch.tensor(127.5199966430664))
    q_input.true_quantize = True
    im = q_input(im)

    x = model(im)


def check_correctness_mbv2():

    from zoo.mobilenet.mobilenet_v2_q import mobilenet_v2, InvertedResidual
    from utils.layer import QAddition, search_turn_off_relu6

    model = mobilenet_v2()
    model.load_state_dict(torch.load('model/pretrained/mobilenet_v2-b0353104.pth'))
    for m in model.modules():
        if isinstance(m, InvertedResidual):
            m.turn_on_add()

    search_absorb_bn(model)
    search_replace_convolution2d(model, 8)
    mark_layer(model, 'root')

    model.load_state_dict(torch.load('result/pth/true_quantized.pth'))

    for m in model.modules():
        if isinstance(m, QConv2d):
            m.use_full_quantization()
            # m.saturated = False
            # m.use_quantization_simulation()
        if isinstance(m, QAddition):
            # m.reset_quantization()
            # m.use_quantization_simulation()
            m.use_full_quantization()
            # m.saturated = False
            # m.forward_1 = False
        if isinstance(m, QAvgPooling):
            m.q_inference = True

    search_turn_off_relu6(model)

    model.classifier[1][1].mul.data = torch.tensor(1.0)
    model.classifier[1][1].shift.data = torch.tensor(0.0)
    model.classifier[1][1].saturated = False

    model.eval()

    import scipy.io as sio
    import numpy as np

    a = sio.loadmat('model/img.mat')['img'].astype(np.float32)

    im = torch.from_numpy(a).permute(2, 0, 1)[None]

    test_process = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                                       ])

    # im = torch.round(test_process(im)[None] * 48.1)

    # im = test_process(im)[None]

    x = model(im)


def check_correctness_mbv2_imagenet():
    test_process = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])
                                       ])

    data_set = ImageFolder(root='/media/xumengmeng/新加卷/Dataset/ImageNet/ILSVRC2012_img_val', transform=test_process)
    loader = dataloader.DataLoader(dataset=data_set, num_workers=8, shuffle=False,
                                   batch_size=128)

    # """ Configure a quantized MobileNet v2 """
    # model = mobilenet_v2()
    # model.load_state_dict(torch.load('model/MobileNet_V2/mobilenet_v2-b0353104.pth'))
    # for m in model.modules():
    #     if isinstance(m, InvertedResidual):
    #         m.turn_on_add()
    #
    # search_absorb_bn(model)
    # search_replace_convolution2d(model, 8)
    # mark_layer(model, 'root')
    #
    # model.load_state_dict(torch.load('result/pth/true_quantized.pth'))

    model = torch.load('result/pth/true_quantized_model.pth')

    for m in model.modules():
        if isinstance(m, QConv2d):
            m.use_full_quantization()
            # m.saturated = False
            # m.use_quantization_simulation()
        if isinstance(m, QAddition):
            # m.reset_quantization()
            # m.use_quantization_simulation()
            m.use_full_quantization()
            # m.saturated = False
            # m.forward_1 = False
            m.q_inference = False
        if isinstance(m, QAvgPooling):
            m.q_inference = True
            # m.q_inference = False

    search_turn_off_relu6(model)

    import numpy as np
    model.classifier[1][1].mul.data = torch.tensor(np.ones([1000])*1)
    model.classifier[1][1].shift.data = torch.tensor(np.ones([1000])*1)
    model.classifier[1][1].saturated = False

    model.eval()

    """ Configuration End """

    validate(loader, model)

def check_correctness_mbv1_imagenet():
    test_process = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                            std=[0.5, 0.5, 0.5])
                                       ])

    data_set = ImageFolder(root='/media/xumengmeng/新加卷/Dataset/ImageNet/ILSVRC2012_img_val', transform=test_process)
    loader = dataloader.DataLoader(dataset=data_set, num_workers=8, shuffle=False,
                                   batch_size=128)

    """ Configure a quantized MobileNet v2 """
    # model = MobileNet_v1()
    # model.load_state_dict(torch.load('model/MobileNet_V1/mobilenet_v1_1.0_224.pth'))
    # for m in model.modules():
    #     if isinstance(m, InvertedResidual):
    #         m.turn_on_add()
    #
    # search_absorb_bn(model)
    # search_replace_convolution2d(model, 8)
    # mark_layer(model, 'root')
    #
    # model.load_state_dict(torch.load('result/pth/true_quantized.pth'))

    model = torch.load('result/pth/true_quantized_model.pth')

    for m in model.modules():
        if isinstance(m, QConv2d):
            m.use_full_quantization()
            # m.saturated = False
            # m.use_quantization_simulation()
        if isinstance(m, QAddition):
            # m.reset_quantization()
            # m.use_quantization_simulation()
            m.use_full_quantization()
            # m.saturated = False
            # m.forward_1 = False
            m.q_inference = False
        if isinstance(m, QAvgPooling):
            m.q_inference = True
            # m.q_inference = False

    search_turn_off_relu6(model)

    # model.classifier[1][1].mul.data = torch.tensor(1.0)
    # model.classifier[1][1].shift.data = torch.tensor(0.0)
    # model.classifier[1][1].saturated = False
    # import numpy as np
    # classifier_mul = np.ones([1001])
    # classifier_shift = np.zeros([1001])
    # model.classifier.mul.data = torch.tensor(classifier_mul)
    # model.classifier.shift.data = torch.tensor(classifier_shift)
    model.classifier.saturated = False

    model.eval()

    """ Configuration End """

    validate(loader, model)

    
def validate(val_loader, model, use_gpu=True):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if use_gpu:
        model = model.cuda()

    with torch.no_grad():
        end = time.time()
        for i, (im, target) in enumerate(val_loader):

            if use_gpu:
                # im = torch.round(im.cuda()*48.1)
                im = torch.round(im.cuda()*115.2)

            im_output = model(im)
            if im_output.shape[1] == 1000:
                i_output = im_output
            if im_output.shape[1] == 1001:
                i_output = im_output[:, 1:1001]
            output = i_output.cpu().detach().view(-1, 1000)

            # measure accuracy and record loss
            acc_1, acc_5 = accuracy(output, target, top_k=(1, 5))

            top1.update(acc_1[0], im.size(0))
            top5.update(acc_5[0], im.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg


def validate_float(val_loader, model, use_gpu=True):
    '''
    float mobilenet_v2 validate function
    '''
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if use_gpu:
        model = model.cuda()

    with torch.no_grad():
        end = time.time()
        for i, (im, target) in enumerate(val_loader):

            if use_gpu:
                im = im.cuda()

            im_output = model(im)
            if im_output.shape[1] == 1001:
                i_output = im_output[:, 1:1001]
            if im_output.shape[1] == 1000:
                i_output = im_output
            output = i_output.cpu().detach().view(-1, 1000)

            # measure accuracy and record loss
            acc_1, acc_5 = accuracy(output, target, top_k=(1, 5))

            top1.update(acc_1[0], im.size(0))
            top5.update(acc_5[0], im.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg

def check_correctness_mbv2_v2():

    from model.mobilenet_v2_q import MobileNetV2, InvertedResidual

    from torchvision.models.mobilenet import MobileNetV2

    from utils.layer import QAddition
    model = MobileNetV2()
    model.load_state_dict(torch.load('model/pretrained/mobilenet_v2-b0353104.pth'))

    search_absorb_bn(model)
    cnt = 0
    # for m in model.modules():
    #     if isinstance(m, InvertedResidual):
    #         if len(m.conv) > 3:
    #             fuse_bn(m.conv[0][0], m.conv[0][1])
    #             fuse_bn(m.conv[1][0], m.conv[1][1])
    #             fuse_bn(m.conv[2], m.conv[3])
    #             cnt += 3
    #         else:
    #             fuse_bn(m.conv[0][0], m.conv[0][1])
    #             fuse_bn(m.conv[1], m.conv[2])
    #             cnt += 2
    # fuse_bn(model.features[0][0], model.features[0][1])

    # fuse_bn(model.features[18][0], model.features[18][1])
    # remove_bn_params(model.features[18][1])
    # model.features[18][1].track_running_stats = False
    # model.features[18][1].affine = False

    # fuse_bn(model.features[18][0], model.features[18][1])

    print(cnt + 2)

    for m in model.modules():
        if isinstance(m, InvertedResidual):
            m.turn_on_add()

    model.eval()

    from PIL import Image
    import scipy.io as sio
    # im = Image.open('model/tmp.jpg')

    a = sio.loadmat('model/im.mat')['img']
    im = Image.fromarray(a)

    test_process = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                                       ])

    # im = torch.round(test_process(im)[None])

    im = test_process(im)[None]

    x = model(im)


if __name__ == "__main__":
    # check_correctness_mbv2()
    # check_correctness_mbv2_imagenet()
    check_correctness_mbv1_imagenet()

