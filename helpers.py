import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

batch_size = 1
transform = transforms.Compose([
    transforms.Scale(600),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
det = dset.CocoDetection(root='data/train2014', transform=transform,
                         annFile='data/annotations/instances_train2014.json')
loader = data.DataLoader(det, batch_size=batch_size, shuffle=True, num_workers=6)
loader = iter(loader)


def load_model(model, name):
    state_dict = torch.load('data/{}_state'.format(name))
    model.load_state_dict(state_dict)


def next_batch():

    # Retrieve next set of examples and initialize var for image data
    images, instances = loader.next()
    images = Variable(images, requires_grad=True).cuda()

    # Create vars
    classes, boxes = [], []
    for instance in instances:

        # Create vars for class ids
        class_var = Variable(instance['category_id'])
        classes.append(class_var)

        # Create vars for bounding boxes
        box_vars = Variable(torch.cat(instance['bbox']).float()).cuda()
        boxes.append(box_vars)

    return images, classes, boxes


def save_model(model, name):
    torch.save(model.state_dict(), 'data/{}_state'.format(name))
    print("")
