import argparse
import helpers
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
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
cross_entropy = nn.CrossEntropyLoss()
smooth_l1 = nn.SmoothL1Loss()

# Keep track of time elapsed and running average for loss
start = time.time()
plot_losses = []
print_loss_total = 0


def update(features, cls_targets, reg_targets):

    # Initialize optimizer
    opt.zero_grad()

    # Predict classes and boxes
    cls_scores, reg_scores = rfcn(features, batch_size=1)

    # Compute loss
    loss = 0.
    for i in range(len(cls_targets)):
        cls_target = cls_targets[i]
        cls_targets[i] = cls_targets[i].repeat(cls_scores.size(0))
        loss += cross_entropy(cls_scores, cls_targets[i].cuda())

        is_bg = torch.equal(cls_target.data.int(), torch.zeros([1]).int())
        if not is_bg:
            reg_targets[i] = reg_targets[i].repeat(reg_scores.size(0))
            loss += smooth_l1(reg_scores, reg_targets[i])

    # Backprop and update if any instances were actually provided
    if len(cls_targets) > 0:
        loss.backward()
        nn.utils.clip_grad_norm(rfcn.parameters(), args.grad_clip)
        opt.step()

    return loss

# Training loop
for epoch in range(args.epochs):
    # Get next batch of training data
    examples, classes, boxes = helpers.next_batch()

    # Run the train step
    total_loss = update(examples, classes, boxes)

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
