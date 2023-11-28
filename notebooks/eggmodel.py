from PIL import Image
from torchvision import transforms
from torchvision.models import shufflenet_v2_x0_5 as ShuffleNet05
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from datetime import datetime
from torch import nn

import numpy as np
import torch

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((256, 256)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


