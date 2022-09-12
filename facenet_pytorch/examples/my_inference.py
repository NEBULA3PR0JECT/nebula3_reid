import sys
import os

curr_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(curr_dir)
sys.path.append('/home/hanoch/notebooks/nebula3_reid')
sys.path.append('/home/hanoch/notebooks/nebula3_reid/facenet_pytorch')

from facenet_pytorch import MTCNN, InceptionResnetV1
# from facenet_pytorch.models import mtcnn, inception_resnet_v1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

rel_path = '/home/hanoch/notebooks/nebula3_reid/facenet_pytorch' # 'nebula3_reid/facenet_pytorch'

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
#### Define MTCNN module
"""
Default params shown for illustration, but not needed. Note that, since MTCNN is a collection of neural nets and other code, the device must be passed in the following way to enable copying of objects when needed internally.

See `help(MTCNN)` for more details.

"""


mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
# Modify model to VGGFace based and resnet
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder(os.path.join(rel_path, 'data/test_images'))
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)


aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True) #post_process=False
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print("distance confusion matrix")
print(pd.DataFrame(dists, columns=names, index=names))