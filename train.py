from __future__ import division
import os
import argparse
import pandas as pd
import numpy as np
import yaml
from trainer import trainer
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from loader import *
from model.attention_swin_unet import SwinAttentionUnet as ViT_seg

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default='./configs/config_skin.yml', help='root path for config')
args = parser.parse_args(args=[])

config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss = np.inf
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
data_path = config['path_to_data']
train_dataset = isic_loader(path_Data=data_path, mode="train", image_size=config['image_size'])
train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size_tr']), shuffle=True)
val_dataset = isic_loader(path_Data=data_path, mode="valid")
val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size_va']), shuffle=False)
test_dataset = isic_loader(path_Data=data_path, mode="test")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# build model
Net = ViT_seg(config).cuda()
iNet = Net.to(device)
if config['pretrained']:
    Net.load_state_dict(torch.load(config['saved_model'], map_location='cpu')['model_weights'])
    best_val_loss = torch.load(config['saved_model'], map_location='cpu')['val_loss']

optimizer = optim.Adam(Net.parameters(), lr=float(config['lr']))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=config['patience'])
criteria = torch.nn.BCELoss()
trainer(config, Net, train_loader, test_loader, optimizer, criteria, device)





