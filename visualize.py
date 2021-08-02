import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

import argparse
import os
from tqdm import tqdm
import csv

from utils import *
from model import *
from dataset import DenoiseDataset
from losses import *

device = 'cuda:0'
data_dir = '/root/kpcn_data/kpcn_data/data'
eps = 0.00316

def draw_patches(data):
  gt = data['target_total'].to(device)[0, :, 32:96, 32:96]
  print(gt.shape)
  for x in range(15):
    for y in range(15):
      gt[:, x*64:x*64+64, y*64:y*64+64] = torch.zeros((1, 3, 64, 64))
  return gt

# train_dir = 

def show_patches():
  dataset = DenoiseDataset(data_dir, 8, 'kpcn', 'train', 1, 'recon',
        use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, pnet_out_size=3)
  dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=False
    )
  i = 0
  x, y = 0, 0
  gt_full_patches = torch.zeros((3, 960, 960)).to(device)
  for data in tqdm(dataloader, leave=False, ncols=70):
    # if x == 0 and y == 0:
    #   y += 1
    #   continue
    # if x == 0 and y == 1:
    #   y += 1
    #   continue
    gt = data['target_total'].to(device)[0, :, 32:96, 32:96] # 3 * 64  64
    gt_full_patches[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(gt)
    # print(gt_full_patches[0, x*64, y*64:y*64+64].shape)
    gt_full_patches[0, x*64, y*64:y*64+64] = torch.ones((64), device=device)
    # print(gt_full_patches[1:, x*64, y*64:y*64+64].shape)
    gt_full_patches[1:, x*64, y*64:y*64+64] = torch.zeros((2, 64), device=device)
    gt_full_patches[0, x*64+63, y*64:y*64+64] = torch.ones((64), device=device)
    gt_full_patches[1:, x*64+63, y*64:y*64+64] = torch.zeros((2, 64), device=device)
    gt_full_patches[0, x*64:x*64+64, y*64] = torch.ones((64), device=device)
    gt_full_patches[1:, x*64:x*64+64, y*64] = torch.zeros((2, 64), device=device)
    gt_full_patches[0, x*64:x*64+64, y*64+63] = torch.ones((64), device=device)
    gt_full_patches[1:, x*64:x*64+64, y*64+63] = torch.zeros((2, 64), device=device)


    y += 1
    if x < 15 and y>=15:
      x += 1
      y = 0

    if x >= 15:
      save_image(gt_full_patches, 'draw_patch/gt_patch_{}.png'.format(i))

      x,y = 0,0
      i += 1


def save_features():
  dataset = DenoiseDataset(data_dir, 8, 'kpcn', 'train', 1, 'recon',
        use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, pnet_out_size=3)
  dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=False
    )
  i = 0
  x, y = 0, 0
  input_image = torch.zeros((3, 960, 960)).to(device)
  gt_image = torch.zeros((3, 960, 960)).to(device)
  diff_rad = torch.zeros((3, 960, 960)).to(device)
  diff_rad_var = torch.zeros((1, 960, 960)).to(device)
  diff_rad_dx = torch.zeros((3, 960, 960)).to(device)
  diff_rad_dy = torch.zeros((3, 960, 960)).to(device)
  spec_rad = torch.zeros((3, 960, 960)).to(device)
  spec_rad_var = torch.zeros((1, 960, 960)).to(device)
  spec_rad_dx = torch.zeros((3, 960, 960)).to(device)
  spec_rad_dy = torch.zeros((3, 960, 960)).to(device)
  normal = torch.zeros((3, 960, 960)).to(device)
  normal_var = torch.zeros((1, 960, 960)).to(device)
  normal_dx = torch.zeros((3, 960, 960)).to(device)
  normal_dy = torch.zeros((3, 960, 960)).to(device)
  depth = torch.zeros((1, 960, 960)).to(device)
  depth_var = torch.zeros((1, 960, 960)).to(device)
  depth_dx = torch.zeros((1, 960, 960)).to(device)
  depth_dy = torch.zeros((1, 960, 960)).to(device)
  albedo_in = torch.zeros((3, 960, 960)).to(device)
  albedo_in_var = torch.zeros((1, 960, 960)).to(device)
  albedo_in_dx = torch.zeros((3, 960, 960)).to(device)
  albedo_in_dy = torch.zeros((3, 960, 960)).to(device)
  for data in tqdm(dataloader, leave=False, ncols=70):
    # print(data['target_total'].shape)

    inputFinal = data['kpcn_diffuse_buffer'] * (data['kpcn_albedo'] + eps) + torch.exp(data['kpcn_specular_buffer']) - 1.0
    gt = data['target_total'].to(device)[0, :, 32:96, 32:96]

    input_image[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(inputFinal[0, :, 32:96, 32:96])
    gt_image[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(gt)
    diff_rad[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,:3,:,:][0, :3, 32:96, 32:96]
    diff_rad_var[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,3,:,:][0, 32:96, 32:96]
    diff_rad_dx[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,4:7,:,:][0, :, 32:96, 32:96]
    diff_rad_dy[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,7:10,:,:][0, :, 32:96, 32:96]
    spec_rad[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_specular_in'][:,:3,:,:][0, :3, 32:96, 32:96]
    spec_rad_var[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_specular_in'][:,3,:,:][0, 32:96, 32:96]
    spec_rad_dx[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_specular_in'][:,4:7,:,:][0, :, 32:96, 32:96]
    spec_rad_dy[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_specular_in'][:,7:10,:,:][0, :, 32:96, 32:96]
    normal[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,10:13,:,:][0, :, 32:96, 32:96]
    normal_var[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,13,:,:][0, 32:96, 32:96]
    normal_dx[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,14:17,:,:][0, :, 32:96, 32:96]
    normal_dy[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,17:20,:,:][0, :, 32:96, 32:96]
    depth[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,20,:,:][0, 32:96, 32:96]
    depth_var[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,21,:,:][0, 32:96, 32:96]
    depth_dx[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,22,:,:][0, 32:96, 32:96]
    depth_dy[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,23,:,:][0, 32:96, 32:96]
    albedo_in[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,24:27,:,:][0, :, 32:96, 32:96]
    albedo_in_var[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,27,:,:][0, 32:96, 32:96]
    albedo_in_dx[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,28:31,:,:][0, :, 32:96, 32:96]
    albedo_in_dy[:, x*64:x*64+64, y*64:y*64+64] = data['kpcn_diffuse_in'][:,31:34,:,:][0, :, 32:96, 32:96]

  

    y += 1
    if x < 15 and y>=15:
      x += 1
      y = 0

    if x >= 15:
      if not os.path.exists('train_data/train{}'.format(i)):
          os.makedirs('train_data/train{}'.format(i))
      save_image(input_image, 'train_data/train{}/noisy.png'.format(i))
      save_image(gt_image, 'train_data/train{}/clean.png'.format(i))
      save_image(diff_rad, 'train_data/train{}/diff_rad.png'.format(i))
      save_image(diff_rad_var, 'train_data/train{}/diff_rad_var.png'.format(i))
      save_image(diff_rad_dx, 'train_data/train{}/diff_rad_dx.png'.format(i))
      save_image(diff_rad_dy, 'train_data/train{}/diff_rad_dy.png'.format(i))
      save_image(spec_rad, 'train_data/train{}/spec_rad.png'.format(i))
      save_image(spec_rad_var, 'train_data/train{}/spec_rad_var.png'.format(i))
      save_image(spec_rad_dx, 'train_data/train{}/spec_rad_dx.png'.format(i))
      save_image(spec_rad_dy, 'train_data/train{}/spec_rad_dy.png'.format(i))
      save_image(normal, 'train_data/train{}/normal.png'.format(i))
      save_image(normal_var, 'train_data/train{}/normal_var.png'.format(i))
      save_image(normal_dx, 'train_data/train{}/normal_dx.png'.format(i))
      save_image(normal_dy, 'train_data/train{}/normal_dy.png'.format(i))
      save_image(depth, 'train_data/train{}/depth.png'.format(i))
      save_image(depth_var, 'train_data/train{}/depth_var.png'.format(i))
      save_image(depth_dx, 'train_data/train{}/depth_dx.png'.format(i))
      save_image(depth_dy, 'train_data/train{}/depth_dy.png'.format(i))
      save_image(albedo_in, 'train_data/train{}/albedo.png'.format(i))
      save_image(albedo_in_var, 'train_data/train{}/albedo_var.png'.format(i))
      save_image(albedo_in_dx, 'train_data/train{}/albedo_dx.png'.format(i))
      save_image(albedo_in_dy, 'train_data/train{}/albedo_dy.png'.format(i))
      x,y = 0,0
      i += 1


# show_patches()
save_features()