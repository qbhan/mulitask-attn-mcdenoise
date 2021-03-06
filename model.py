import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

recon_kernel_size = 21

import itertools
import os

from basic_models import se_layer, eca_layer, cbam, simple_feat_layer, spc_layer, nlblock, sablock
from unet import DenseUnet, encoderUnet, decoderUnet, sampleUnet


def make_net(n_layers, input_channels, hidden_channels, kernel_size, mode):
  # create first layer manually
  layers = [
      nn.Conv2d(input_channels, hidden_channels, kernel_size),
      nn.ReLU()
  ]
  
  for l in range(n_layers-2):
    layers += [
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size),
        nn.ReLU()
    ]
    
    # params = sum(p.numel() for p in layers[-2].parameters() if p.requires_grad)
    # print("Params : {}".format(params))
    
  out_channels = 3 if 'dpcn' in mode else recon_kernel_size**2
  layers += [nn.Conv2d(hidden_channels, out_channels, kernel_size)]#, padding=18)]
  
  for layer in layers:
    if isinstance(layer, nn.Conv2d):
      nn.init.xavier_uniform_(layer.weight)
  
  return nn.Sequential(*layers)


def make_recon_net(n_layers, input_channels, hidden_channels, kernel_size, mode):
  # create first layer manually
  padding = 1 if kernel_size == 3 else 2
  layers = [
      nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding),
      nn.ReLU()
  ]
  
  for l in range(n_layers-2):
    layers += [
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding),
        nn.ReLU()
    ]
    
    # params = sum(p.numel() for p in layers[-2].parameters() if p.requires_grad)
    # print("Params : {}".format(params))
    
  out_channels = 3 if 'dpcn' in mode else recon_kernel_size**2
  layers += [nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=padding)]#, padding=18)]

  if 'dpcn' in mode:
    layers += [nn.Sigmoid()]
  
  for layer in layers:
    if isinstance(layer, nn.Conv2d):
      nn.init.xavier_uniform_(layer.weight)
  
  return nn.Sequential(*layers)


def make_senet(n_layers, input_channels, hidden_channels, kernel_size, mode):
  # create first layer manually
  layers = [
      nn.Conv2d(input_channels, hidden_channels, kernel_size),
      nn.ReLU(),
      se_layer(hidden_channels, reduction=8)
  ]
  
  for l in range(n_layers-2):
    layers += [
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size),
        nn.ReLU(),
        se_layer(hidden_channels, reduction=8)
    ]
    
    params = sum(p.numel() for p in layers[-2].parameters() if p.requires_grad)
    # print("Params : {}".format(params))
    
  out_channels = 3 if 'dpcn' in mode else recon_kernel_size**2
  layers += [nn.Conv2d(hidden_channels, out_channels, kernel_size)]#, padding=18)]
  
  for layer in layers:
    if isinstance(layer, nn.Conv2d):
      nn.init.xavier_uniform_(layer.weight)
  
  return nn.Sequential(*layers)


def make_ecanet(n_layers, input_channels, hidden_channels, kernel_size, mode):
  # create first layer manually
  layers = [
      nn.Conv2d(input_channels, hidden_channels, kernel_size),
      nn.ReLU(),
      eca_layer()
  ]
  
  for l in range(n_layers-2):
    layers += [
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size),
        nn.ReLU(),
        eca_layer()
    ]
    
    params = sum(p.numel() for p in layers[-2].parameters() if p.requires_grad)
    # print("Params : {}".format(params))
    
  out_channels = 3 if 'dpcn' in mode else recon_kernel_size**2
  layers += [nn.Conv2d(hidden_channels, out_channels, kernel_size)]#, padding=18)]
  
  for layer in layers:
    if isinstance(layer, nn.Conv2d):
      nn.init.xavier_uniform_(layer.weight)
  
  return nn.Sequential(*layers)


def make_cbamnet(n_layers, input_channels, hidden_channels, kernel_size, mode):
  # create first layer manually
  layers = [
      nn.Conv2d(input_channels, hidden_channels, kernel_size),
      nn.ReLU(),
      cbam(hidden_channels, reduction_ratio=4)
  ]
  
  for l in range(n_layers-2):
    layers += [
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size),
        nn.ReLU(),
        cbam(hidden_channels, reduction_ratio=4)
    ]
    
    params = sum(p.numel() for p in layers[-2].parameters() if p.requires_grad)
    # print("Params : {}".format(params))
    
  out_channels = 3 if 'dpcn' in mode else recon_kernel_size**2
  layers += [nn.Conv2d(hidden_channels, out_channels, kernel_size)]#, padding=18)]
  
  for layer in layers:
    if isinstance(layer, nn.Conv2d):
      nn.init.xavier_uniform_(layer.weight)
  
  return nn.Sequential(*layers)


def make_simple_feat_net(n_layers, input_channels, hidden_channels, kernel_size, mode, branch='diff'):
  # create first layer manually
  print('make simple feat')
  layers = [
      # simple_feat_layer(input_channels, branch + '_0', debug=False),
      nn.Conv2d(input_channels, hidden_channels, kernel_size),
      nn.ReLU(),
      simple_feat_layer(hidden_channels, branch + '_1', debug=False)
  ]
  
  for l in range(n_layers-2):
    layers += [
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size),
        nn.ReLU(),
        simple_feat_layer(hidden_channels, branch + '_' + str(2+l), debug=False)
    ]
    
    params = sum(p.numel() for p in layers[-2].parameters() if p.requires_grad)
    # print("Params : {}".format(params))
    
  out_channels = 3 if 'dpcn' in mode else recon_kernel_size**2
  layers += [nn.Conv2d(hidden_channels, out_channels, kernel_size)]#, padding=18)]
  
  for layer in layers:
    if isinstance(layer, nn.Conv2d):
      nn.init.xavier_uniform_(layer.weight)
  
  return nn.Sequential(*layers)


# def crop_like(data, like, debug=False):
#   if data.shape[-2:] != like.shape[-2:]:
#     # crop
#     with torch.no_grad():
#       dx, dy = data.shape[-2] - like.shape[-2], data.shape[-1] - like.shape[-1]
#       data = data[:,:,dx//2:-dx//2,dy//2:-dy//2]
#       if debug:
#         print(dx, dy)
#         print("After crop:", data.shape)
#   return data


def make_simple_net(n_layers, input_channels, hidden_channels, kernel_size, mode, branch='diff'):
  print('make simple')
  layers = [
      simple_feat_layer(input_channels, branch + '_0', debug=False),
      nn.Conv2d(input_channels, hidden_channels, kernel_size),
      nn.ReLU(),
      # simple_feat_layer(hidden_channels, branch + '_1', debug=True)
  ]
  
  for l in range(n_layers-2):
    layers += [
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size),
        nn.ReLU(),
        # simple_feat_layer(hidden_channels, branch + '_' + str(2+l), debug=True)
    ]
    
    params = sum(p.numel() for p in layers[-2].parameters() if p.requires_grad)
    # print("Params : {}".format(params))
    
  out_channels = 3 if 'dpcn' in mode else recon_kernel_size**2
  layers += [nn.Conv2d(hidden_channels, out_channels, kernel_size)]#, padding=18)]
  
  for layer in layers:
    if isinstance(layer, nn.Conv2d):
      nn.init.xavier_uniform_(layer.weight)
  
  return nn.Sequential(*layers)


def make_DenseUnet(input_channels, mode):
  return DenseUnet(input_channels, mode)


def make_spc_net(n_layers, input_channels, hidden_channels, kernel_size, mode, branch='diff'):
  # create first layer manually
  print('make spc feat')
  layers = [
      # spc_layer(input_channels, 0, debug=True),
      nn.Conv2d(input_channels, hidden_channels, kernel_size),
      nn.ReLU(),
      spc_layer(hidden_channels, branch + '_' + str(1), debug=True)
  ]
  
  for l in range(n_layers-2):
    layers += [
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size),
        nn.ReLU(),
        spc_layer(hidden_channels, branch + '_' + str(l+2), debug=True)
    ]
    
    params = sum(p.numel() for p in layers[-2].parameters() if p.requires_grad)
    # print("Params : {}".format(params))
    
  out_channels = 3 if 'dpcn' in mode else recon_kernel_size**2
  layers += [nn.Conv2d(hidden_channels, out_channels, kernel_size)]#, padding=18)]
  
  for layer in layers:
    if isinstance(layer, nn.Conv2d):
      nn.init.xavier_uniform_(layer.weight)
  
  return nn.Sequential(*layers)


def make_nlb_net(n_layers, input_channels, hidden_channels, kernel_size, mode, branch='diff'):
  # create first layer manually
  print('make nonlocalnet')
  layers = [
      # nlblock(input_channels, input_channels//2),
      nn.Conv2d(input_channels, hidden_channels, kernel_size),
      nn.ReLU(),
  ]
  
  for l in range(n_layers-2):
    layers += [
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size),
        nn.ReLU(),
    ]
    
    params = sum(p.numel() for p in layers[-2].parameters() if p.requires_grad)
    # print("Params : {}".format(params))
    
  out_channels = 3 if 'dpcn' in mode else recon_kernel_size**2
  layers += [nlblock(hidden_channels, hidden_channels//2), nn.Conv2d(hidden_channels, out_channels, kernel_size)]#, padding=18)]
  
  for layer in layers:
    if isinstance(layer, nn.Conv2d):
      nn.init.xavier_uniform_(layer.weight)
  
  return nn.Sequential(*layers)


def apply_kernel(weights, data, device):
    # print('WEIGHTS: {}, DATA : {}'.format(weights.shape, data.shape))
    # apply softmax to kernel weights
    # print(weights.shape)
    weights = weights.permute((0, 2, 3, 1)).to(device)
    # print(weights.shape, data.shape)
    _, _, h, w = data.size()
    weights = F.softmax(weights, dim=3).view(-1, w * h, recon_kernel_size, recon_kernel_size)
    # print(weights.shape, data.shape)
    # now we have to apply kernels to every pixel
    # first pad the input
    r = recon_kernel_size // 2
    data = F.pad(data[:,:3,:,:], (r,) * 4, "reflect")
    # print(data.shape)
    #print(data[0,:,:,:])
    
    # make slices
    R = []
    G = []
    B = []
    kernels = []
    for i in range(h):
      for j in range(w):
        pos = i*h+j
        # ws = weights[:,pos:pos+1,:,:]
        # kernels += [ws, ws, ws]
        sy, ey = i+r-r, i+r+r+1
        sx, ex = j+r-r, j+r+r+1
        R.append(data[:,0:1,sy:ey,sx:ex])
        G.append(data[:,1:2,sy:ey,sx:ex])
        B.append(data[:,2:3,sy:ey,sx:ex])
        #slices.append(data[:,:,sy:ey,sx:ex])
        
    reds = (torch.cat(R, dim=1).to(device)*weights).sum(2).sum(2)
    greens = (torch.cat(G, dim=1).to(device)*weights).sum(2).sum(2)
    blues = (torch.cat(B, dim=1).to(device)*weights).sum(2).sum(2)
    
    res = torch.cat((reds, greens, blues), dim=1).view(-1, 3, h, w).to(device)
    # print(res.shape)
    
    return res


def make_sa_net(n_layers, input_channels, hidden_channels, kernel_size, mode, branch='diff'):
  # create first layer manually
  print('make spc feat')
  layers = [
      # spc_layer(input_channels, 0, debug=True),
      nn.Conv2d(input_channels, hidden_channels, kernel_size),
      nn.ReLU(),
      sablock(hidden_channels, hidden_channels)
  ]
  
  for l in range(n_layers-2):
    layers += [
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size),
        nn.ReLU(),
        sablock(hidden_channels, hidden_channels)
    ]
    
    params = sum(p.numel() for p in layers[-2].parameters() if p.requires_grad)
    # print("Params : {}".format(params))
    
  out_channels = 3 if 'dpcn' in mode else recon_kernel_size**2
  layers += [nn.Conv2d(hidden_channels, out_channels, kernel_size)]#, padding=18)]
  
  for layer in layers:
    if isinstance(layer, nn.Conv2d):
      nn.init.xavier_uniform_(layer.weight)
  
  return nn.Sequential(*layers)


# spc = make_spc_net(9, 34, 100, 5, 'spc_kpcn')
# simple = make_simple_feat_net(9, 34, 100, 5, 'simple_feat_kpcn')
# print('# Parameter for spcnet : {}'.format(sum([p.numel() for p in spc.parameters()])))
# print('# Parameter for simplefeatNet : {}'.format(sum([p.numel() for p in simple.parameters()])))