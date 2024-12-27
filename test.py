from __future__ import division
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
from loader import *
import numpy as np
import yaml
from model.attention_swin_unet import SwinAttentionUnet as ViT_seg
import argparse
from inference import inference
from torchinfo import summary  #
import time #
import io #
from contextlib import redirect_stdout #
from thop import clever_format, profile


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str,default='./configs/config_skin.yml', help='root path for config')
args = parser.parse_args(args=[])
config = yaml.load(open(args.config_path), Loader=yaml.FullLoader)

number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss  = np.inf
patience       = 0
data_path = config['path_to_data']
test_dataset = isic_loader(path_Data = data_path,mode = "test")
test_loader  = DataLoader(test_dataset, batch_size = 1)

print("config:",config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Net = ViT_seg(config).to(device)

model_weights = torch.load(config["saved_model"], map_location='cpu')['model_weights']
Net.load_state_dict(model_weights)

batch = next(iter(test_loader))
img = batch['image']
img = img.float()
input_size = tuple(img.size())

f = io.StringIO()

with redirect_stdout(f):
    model_summary = summary(Net, input_size, depth=3)

print(model_summary)

img = img.to(device).float()
flops, params = profile(Net, inputs=(img, ))
flops, params = clever_format([flops, params], "%.3f")
print(f"FLOPs: {flops}, Params: {params}")

start_time = time.time()

inference(Net, test_loader)

end_time = time.time()

total_inference_time = end_time - start_time

total_samples = len(test_loader.dataset)

average_inference_time_per_sample = total_inference_time / total_samples

print("Average inference time per sample: {:.6f} seconds".format(average_inference_time_per_sample))

out = f.getvalue()

print(out)

with open("model_summary.txt", "w") as file:
    file.write(out)
    file.write(f"Total FLOPs: {flops}\n")
    file.write(f"Inference time: {end_time - start_time:.2f} seconds\n")


