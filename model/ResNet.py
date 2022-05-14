import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        # 预处理层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.Hair_Classfier = nn.Sequential(nn.Linear(512,256),nn.Linear(256,128),nn.Linear(128,2))
        self.Gender_Classfier = nn.Sequential(nn.Linear(512,256),nn.Linear(256,128),nn.Linear(128,2))
        self.Ear_Classfier = nn.Sequential(nn.Linear(512,256),nn.Linear(256,128),nn.Linear(128,2))
        self.Smile_Classfier = nn.Sequential(nn.Linear(512,256),nn.Linear(256,128),nn.Linear(128,2))
        self.Frontal_Classfier = nn.Sequential(nn.Linear(512,256),nn.Linear(256,128),nn.Linear(128,2))
        self.Style_Classfier = nn.Sequential(nn.Linear(512,256),nn.Linear(256,128),nn.Linear(128,3))
    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))#[b,3,H,W] => [b,64,?,?]
        # [b, 64, h, w] => [b, 1024, h, w]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # [b, 512, 16, 16] => [b, 512, 1, 1]
        x = self.avgpool(x)
        #print("after pool:", x.shape) # [b, 512, 1, 1]
        # [b, 512, 1, 1] => [b, 512]
        x = x.view(x.size(0), -1)
        Hair_preds = self.Hair_Classfier(x).unsqueeze(0)
        Gender_preds = self.Gender_Classfier(x).unsqueeze(0)
        Ear_preds = self.Ear_Classfier(x).unsqueeze(0)
        Smile_preds = self.Smile_Classfier(x).unsqueeze(0)
        Frontal_preds = self.Frontal_Classfier(x).unsqueeze(0)
        preds = torch.cat((Hair_preds,Gender_preds,Ear_preds,Smile_preds,Frontal_preds),dim=0)
        #[5,b,2]
        Style_preds = self.Style_Classfier(x)
        #[b,3]
        return F.log_softmax(preds,dim=1),F.log_softmax(Style_preds,dim=1)
def forward(self,x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    Hair_preds = self.Hair_Classfier(x).unsqueeze(0)
    Gender_preds = self.Gender_Classfier(x).unsqueeze(0)
    Ear_preds = self.Ear_Classfier(x).unsqueeze(0)
    Smile_preds = self.Smile_Classfier(x).unsqueeze(0)
    Frontal_preds = self.Frontal_Classfier(x).unsqueeze(0)
    preds = torch.cat((Hair_preds, Gender_preds, Ear_preds, Smile_preds, Frontal_preds), dim=0)
    # [5,b,2]
    Style_preds = self.Style_Classfier(x)
    # [b,3]
    return F.log_softmax(preds, dim=1), F.log_softmax(Style_preds, dim=1)
def Load_Model_Resnet18(pretrained=False):
    if pretrained:
        model = models.resnet18(pretrained=True)
        #innum = model.fc.in_features
        #model.fc = nn.Linear(innum, 256)
        model.Hair_Classfier = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 2))
        model.Gender_Classfier = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 2))
        model.Ear_Classfier = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 2))
        model.Smile_Classfier = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 2))
        model.Frontal_Classfier = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 2))
        model.Style_Classfier = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 3))
        model.forward = forward
        return model
    else:
        model = ResNet18()
        return model
