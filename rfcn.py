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
        self.avg_pool = nn.AvgPool3d(self.k)
        self.cls_vote = nn.Softmax2d()

    def forward(self, x, batch_size=8):

        # Get region proposals
        x_cls, x_reg = self.rpn(x)

        # Class voting
        x_cls = self.pos_sensitive_roi_pool(x_cls, self.c + 1, self.k, batch_size)
        x_cls = self.cls_vote(x_cls.squeeze(-1))

        # Box regression
        x_reg = self.pos_sensitive_roi_pool(x_reg.squeeze(-1).squeeze(-1), 4, self.k, batch_size)

        return x_cls, x_reg

    def pos_sensitive_roi_pool(self, x, num_maps, kernel_size, batch_size, kind):
        x = x.view(batch_size, num_maps, -1, kernel_size, kernel_size)
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
        self.reduce_dim = nn.Conv2d(2048, 1024, kernel_size=1, padding=(1, 1))
        self.slide_cls = nn.Conv2d(1024, self.c + 1, kernel_size=self.k, padding=(1, 1))
        self.slide_reg = nn.Conv2d(1024, 4, kernel_size=self.k, padding=(1, 1))

    def forward(self, x):
        # Compute feature maps to slide over
        for l in self.backbone_layers:
            x = getattr(self.backbone_net, l)(x)
        x = self.reduce_dim(x)

        # Slide over maps and compute proposed regions
        x_cls = self.slide_cls(x)
        x_reg = self.slide_reg(x)

        return x_cls, x_reg
