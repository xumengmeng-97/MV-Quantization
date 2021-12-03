from zoo.mobilenet_v1.mobilenet_pytorch import MobileNet_v1
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from dev import validate_float
import torchvision.transforms as transforms
from torchvision import utils
import torch.utils.model_zoo as  model_zoo
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

traindir = '/media/xumengmeng/新加卷/Dataset/ImageNet/ILSVRC2012_img_train/'
valdir = '/media/xumengmeng/新加卷/Dataset/ImageNet/ILSVRC2012_img_val/'
input_size = 224
n_worker = 16
batch_size = 64
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

train_dataset = torchvision.datasets.ImageFolder(
    root=traindir,
    transform=train_transform
    )

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=n_worker, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataset= torchvision.datasets.ImageFolder(root=valdir,
                                              transform=transforms.Compose([
                                                  transforms.Resize(256),
                                                  transforms.CenterCrop(input_size),
                                                  transforms.ToTensor(),
                                                  normalize
                                              ])),
    batch_size=batch_size, shuffle=False,
    num_workers=n_worker, pin_memory=True)

def train(epoch, net):
    running_loss = 0.0
    net = net.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %3f'%(epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0
    return net

def test(model):
    correct = 0
    total = 0
    model.eval()
    model = model.cuda()
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            predicted_ = predicted - 1
            total += labels.size(0)
            correct += (predicted_ ==labels).sum().item()
        print('Accuracy on test set : %.3f %% : %d  %d'%(100*correct / total, correct, total))

if __name__ == '__main__':
    # net = torch.load('./MobileNet_v2_pretrained.pth')
    net = MobileNet_v1()
    net.load_state_dict(torch.load('./model/MobileNet_V1/mobilenet_v1_1.0_224.pth'))
    from torchsummary import summary
    net.cuda()
    summary(net, input_size=(3, 224, 224))

    # net.load_state_dict(torch.load('./mobilenet_v2-b0353104.pth'))
    validate_float(val_loader, net)
    test(net)