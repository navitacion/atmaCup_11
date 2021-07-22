import torchvision.models
from torch import nn
import torch
# Pytorch Image Model
# https://rwightman.github.io/pytorch-image-models/
import timm
from timm import create_model


class Timm_model(nn.Module):
    def __init__(self, model_name, pretrained=False, out_dim=1, mode='regression'):
        super(Timm_model, self).__init__()
        self.base = create_model(model_name, pretrained=pretrained)
        self.mode = mode

        if 'efficientnet' in model_name:
            self.base.classifier = nn.Linear(in_features=self.base.classifier.in_features, out_features=out_dim)
        elif 'vit' in model_name:
            self.base.head = nn.Linear(in_features=self.base.head.in_features, out_features=out_dim)
        else:
            self.base.fc = nn.Linear(in_features=self.base.fc.in_features, out_features=out_dim)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.base(x)

class Torch_model(nn.Module):
    def __init__(self, model_name, pretrained=False, out_dim=1):
        super(Torch_model, self).__init__()
        if model_name == 'resnet18':
            self.base = torchvision.models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.base = torchvision.models.resnet34(pretrained=pretrained)
        if model_name == 'resnet50':
            self.base = torchvision.models.resnet50(pretrained=pretrained)

        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    # Print Timm Models
    model_names = timm.list_models(pretrained=True)
    print(sorted(list(model_names)))

    # net = Timm_model('resnet34d')
    #
    # for name, param in net.named_parameters():
    #     print(name)
    #
    # print(net)

    # inp = torch.randn(4, 3, 224, 224)
    # net = Torch_model(model_name='resnet50')
    #
    # out = net(inp)
    # print(out.size())