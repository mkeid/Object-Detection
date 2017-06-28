import argparse
import helpers
import time
import torch.nn as nn
from rpn import RegionProposalNet
from rfcn import RegionBasedFullyConvNet

# Parse args
parser = argparse.ArgumentParser()
args = parser.parse_args()

# Initialize models
rpn = RegionProposalNet()
rfcn = RegionBasedFullyConvNet()

# Move models to GPU
rpn.cuda()
rfcn.cuda()
