import argparse
import helpers
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from rfcn import RegionBasedFCN


# Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=20000)
parser.add_argument('--grad_clip', default=10.)
parser.add_argument('--learning_rate', default=.001)
parser.add_argument('--print_every', default=100)
parser.add_argument('--save_every', default=1000)
args = parser.parse_args()

# Initialize models

rfcn = RegionBasedFCN().cuda()

# Initialize optimizers and criterion
opt = optim.Adam(rfcn.parameters(), lr=args.learning_rate, weight_decay=.0005)
criterion = nn.NLLLoss()

# Keep track of time elapsed and running average for loss
start = time.time()
plot_losses = []
print_loss_total = 0


def update(features, targets):
    # Initialize optimizer and loss
    opt.zero_grad()
    loss = 0.

    #
    cls_scores, reg_scores = rfcn(features)

    # Backprop and update
    loss.backward()
    nn.utils.clip_grad_norm(rfcn.parameters(), args.grad_clip)
    opt.step()

    return loss

# Training loop
for epoch in range(args.epochs):
    # Get next batch of training data
    examples, labels = helpers.next_batch()

    # Run the train step
    loss = update(examples, labels)

    # Prevent premature logging
    if epoch == 0:
        continue

    # Print training status
    if epoch % args.print_every == 0:
        print("")

    # Save models
    if epoch % args.save_every == 0:
        helpers.save_model(rfcn, 'rfcn')

helpers.save_model(rfcn, 'rfcn')
