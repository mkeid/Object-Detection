import torch.nn as nn
import torchvision.models as models


class RegionBasedFCN(nn.Module):

    def __init__(self, k=3, c=90):
        super(RegionBasedFCN, self).__init__()

        # Keep parameters
        self.k = k
        self.c = c

        # Initialize region proposal network
        self.rpn = RegionProposalNet(self.k, self.c)

        # Initialize layers
        self.roi_pool = nn.AvgPool2d(self.k * self.k)
        self.cls_vote = nn.Softmax2d()

    def forward(self, x, batch_size=8):

        # Compute RoIs [batch_size, k * k * c, height, width]
        x_cls, x_reg = self.rpn(x, batch_size=batch_size)

        # Pool RoIs for class voting [batch_size, n_roi, c]
        x_cls = x_cls.view(batch_size, self.c, self.k * self.k, -1)
        x_cls = self.roi_pool(x_cls).view(batch_size, -1, 1, self.c)
        x_cls = self.cls_vote(x_cls).squeeze(-2)
        x_cls = x_cls.squeeze(0)

        # Pool RoIs for box regression [batch_size, n_roi, 4]
        x_reg = x_reg.view(batch_size, 4, self.k * self.k, -1)
        x_reg = self.roi_pool(x_reg).view(batch_size, -1, 4)
        x_reg = x_reg.squeeze(0)

        return x_cls, x_reg


class RegionProposalNet(nn.Module):

    def __init__(self, k=3, c=21):
        super(RegionProposalNet, self).__init__()

        # Keep parameters
        self.k = k
        self.c = c

        # Initialize image encoder
        self.backbone_net = models.resnet101(pretrained=True)
        self.backbone_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

        # Regional proposal layers
        self.reduce_dim = nn.Conv2d(2048, 1024, kernel_size=1, padding=(1, 1))
        self.slider_cls = nn.Conv2d(1024, self.k * self.k * self.c, kernel_size=self.k, padding=(1, 1))
        self.slider_reg = nn.Conv2d(1024, self.k * self.k * 4, kernel_size=self.k, padding=(1, 1))

    def forward(self, x, batch_size=8):

        # Compute feature maps to slide over
        for l in self.backbone_layers:
            x = getattr(self.backbone_net, l)(x)
        x = self.reduce_dim(x)

        # Slide over maps and compute proposed regions
        x_cls = self.slider_cls(x)
        x_reg = self.slider_reg(x)

        return x_cls, x_reg
