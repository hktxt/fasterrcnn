import os
import torch
import numpy as np
from utils.dataset import LoadDataset
from utils import torch_utils
from tqdm import tqdm, tnrange, tqdm_notebook


# specify visible GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3,4,5'
device = torch_utils.select_device()

torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

path = 'E:/data/voc07/VOCdevkit/VOC2007'

train_set = LoadDataset(path, split='trainval', train=True)
test_set = LoadDataset(path, split='test', train=False)

