from PIL import Image
from torchvision import transforms
from torchvision.models import shufflenet_v2_x0_5 as ShuffleNet05
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from datetime import datetime
from torch import nn

import numpy as np
import torch

preprocess = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((256, 256)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = ImageFolder("eggs", preprocess) # ~6000
dataset_test = ImageFolder("eggs_test", preprocess) # ~600

train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset_test, batch_size=64, shuffle=True)




