import torch
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

batch_size = 8
transform = transforms.Compose([
    transforms.Scale(600),
    transforms.CenterCrop(600),
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
    images, keypoints = loader.next()
    images = Variable(images).cuda()
    return images, keypoints


def save_model(model, name):
    torch.save(model.state_dict(), 'data/{}_state'.format(name))
    print("")
