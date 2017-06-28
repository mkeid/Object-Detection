import torch.nn as nn
import torchvision.models as models


class RegionBasedFCN(nn.Module):
    def __init__(self, k=3, c=20):
        super(RegionBasedFCN, self).__init__()

        # Keep parameters
        self.k = k
        self.c = c

        # Initialize region proposal network
        self.rpn = RegionProposalNet(self.k, self.c)

        # Initialize layers
        self.avg_pool = nn.AvgPool3d(self.rpn.k)
        self.cls_vote = nn.Softmax2d()

    def forward(self, x, batch_size=8):

        # Get region proposals
        x_cls, x_reg = self.rpn(x)

        # Position-sensitive pooling of maps
        x_cls = self.pos_sensitive_roi_pool(x_cls, batch_size)
        x_cls = self.cls_vote(x_cls.squeeze(-1))

        # Box regression
        x_reg = self.pos_sensitive_roi_pool(x_reg, batch_size)
        x_reg = self.avg_pool(x_reg)

        return x_cls, x_reg

    def pos_sensitive_roi_pool(self, x, batch_size):
        x = x.view(batch_size, self.c + 1, -1, self.k, self.k)
        return self.avg_pool(x)


class RegionProposalNet(nn.Module):
    def __init__(self, k=3, c=20):
        super(RegionProposalNet, self).__init__()

        # Keep parameters
        self.k = k
        self.c = c

        # Initialize image encoder
        self.backbone_net = models.resnet101(pretrained=True)
        self.backbone_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']

        # Regional proposal layers
        self.reduce_dim = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1)
        self.slide_cls = nn.Conv2d(in_channels=1024, out_channels=self.c + 1, kernel_size=self.k)
        self.slide_reg = nn.Conv2d(in_channels=self.c + 1, out_channels=4, kernel_size=self.k)

    def forward(self, x):
        # Compute feature maps to slide over
        for l in self.backbone_layers:
            x = getattr(self.backbone_net, l)(x)
        x = self.reduce_dim(x)

        # Slide over maps and compute proposed regions
        x_cls = self.slide_cls(x)
        x_reg = self.slide_reg(x)

        return x_cls, x_reg
