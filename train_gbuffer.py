import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import csv
import random
from tqdm import tqdm

from utils import *
from model import *
from unet import DenseUnet, encoderUnet, decoderUnet, sampleUnet
from losses import *
from dataset import MSDenoiseDataset, init_data

# from test_cython import *

# L = 9 # number of convolutional layers
# n_kernels = 100 # number of kernels in each layer
# kernel_size = 5 # size of kernel (square)

# # input_channels = dataset[0]['X_diff'].shape[-1]
# hidden_channels = 100

permutation = [0, 3, 1, 2]
eps = 0.00316

parser = argparse.ArgumentParser(description='Train the model')

'''
Needed parameters
1. Data & Model specifications
device : which device will the data & model should be loaded
mode : which kind of model should it train
input_channel : input channel
hidden_channel : hidden channel
num_layer : number of layers / depth of models
'''
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--mode', default='kpcn')
parser.add_argument('--num_layers', default=9, type=int)
parser.add_argument('--input_channels', default=34, type=int)
parser.add_argument('--hidden_channels', default=100, type=int)
parser.add_argument('--kernel_size', default=5, type=int)

'''
2. Preprocessing specifications
eps
'''
parser.add_argument('--eps', default=0.00316, type=float)

'''
3. Training Specification
val : should it perform validation
early_stopping : should it perform early stopping
trainset : dataset for training
valset : dataset for validation
lr : learning rate
epoch : epoch
criterion : which loss function should it use
'''
parser.set_defaults(do_feature_dropout=False)
parser.add_argument('--do_feature_dropout', dest='do_feature_dropout', action='store_true')
parser.set_defaults(do_finetune=False)
parser.add_argument('--do_finetune', dest='do_finetune', action='store_true')
parser.set_defaults(do_val=False)
parser.add_argument('--do_val', dest='do_val', action='store_true')
parser.set_defaults(do_early_stopping=False)
parser.add_argument('--do_early_stopping', dest='do_early_stopping', action='store_true')
parser.add_argument('--data_dir')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--loss', default='L1')

save_dir = 'multitask_albedo_scale_1_12'
writer = SummaryWriter('runs/'+save_dir)


def feature_dropout(tensor_input, threshold, device):
  r = random.random()
  if r < (threshold / 3):
    size = tensor_input[:,10:20,:,:].shape
    print(size)
    tensor_input[:,10:20,:,:] = torch.zeros(size, device=device)
  elif r < (2 * threshold / 3):
    size = tensor_input[:,20:24,:,:].shape
    print(size)
    tensor_input[:,20:24,:,:] = torch.zeros(size, device=device)
  elif r < threshold:
    size = tensor_input[:,24:34,:,:].shape
    print(size)
    tensor_input[:,24:34,:,:] = torch.zeros(size, device=device)
  return tensor_input



def validation(models, dataloader, eps, criterion, device, epoch, mode='kpcn'):
  pass
  lossDiff = 0
  lossSpec = 0
  lossFinal = 0
  relL2Final = 0
  lossAlbedo = 0
  lossDepth = 0
  lossNormal = 0
  relL2 = RelativeMSE()
  # for batch_idx, data in enumerate(dataloader):
  batch_idx = 0
  # diffuseNet, specularNet = diffuseNet.eval(), specularNet.eval()

  diffuseNet, specularNet = models['diffuse'].to(device), models['specular'].to(device)
  encodeNet = models['encode'].to(device)
  albedoNet, depthNet, normalNet = models['albedo'].to(device), models['depth'].to(device), models['normal'].to(device)
  diffuseNet.eval(), specularNet.eval(), encodeNet.eval(), albedoNet.eval(), depthNet.eval(), normalNet.eval()
  with torch.no_grad():
    for data in tqdm(dataloader, leave=False, ncols=70):
      X_diff = data['kpcn_diffuse_in'].to(device)
      Y_diff = data['target_diffuse'].to(device)

      # print(diffuseNet.layer)

      # if batch_idx == 10:
      #   pass
      #   diffuseNet.

      outputDiff = diffuseNet(X_diff)
      # if mode == 'KPCN':
      if 'kpcn' in mode:
        X_input = crop_like(X_diff, outputDiff)
        outputDiff = apply_kernel(outputDiff, X_input, device)

      Y_diff = crop_like(Y_diff, outputDiff)
      lossDiff += criterion(outputDiff, Y_diff).item()

      X_spec = data['kpcn_specular_in'].to(device)
      Y_spec = data['target_specular'].to(device)
      
      outputSpec = specularNet(X_spec)
      # if mode == 'KPCN':
      if 'kpcn' in mode:
        X_input = crop_like(X_spec, outputSpec)
        outputSpec = apply_kernel(outputSpec, X_input, device)

      Y_spec = crop_like(Y_spec, outputSpec)
      lossSpec += criterion(outputSpec, Y_spec).item()

      # calculate final ground truth error
      albedo = data['kpcn_albedo'].to(device)
      albedo = crop_like(albedo, outputDiff)
      outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0
      # print('OUTPUT SIZE : {}'.format(outputDiff.shape))
      Y_final = data['target_total'].to(device)
      Y_final = crop_like(Y_final, outputFinal)
      lossFinal += criterion(outputFinal, Y_final).item()
      relL2Final += relL2(outputFinal, Y_final)


      normal = crop_like(data['kpcn_diffuse_in'][:,10:13,:,:], outputDiff).to(device)
      depth = crop_like(data['kpcn_diffuse_in'][:,20,:,:], outputDiff).to(device)

      # g-buffer estimation
      enc = encodeNet(outputFinal) # 8 * 3 * 96 * 96
      # print(len(enc))
      outputAlbedo, outputDepth, outputNormal = albedoNet(enc[0], enc[1], enc[2], enc[3]), depthNet(enc[0], enc[1], enc[2], enc[3]), normalNet(enc[0], enc[1], enc[2], enc[3])
      # outputAlbedo, outputDepth, outputNormal = crop_like(albedo, )
      # print('Depth SIZE : {}, {}'.format(outputDepth.shape, depth.unsqueeze(1).shape))
      lossAlbedo +=  criterion(outputAlbedo, albedo).item()
      lossDepth += criterion(outputDepth, depth.unsqueeze(1)).item()
      lossNormal += criterion(outputNormal, normal).item()
      # loss_finetune = lossFinal + lossAlbedo, lossDepth, lossNormal


      # visualize
      if batch_idx == 10:
        inputFinal = data['kpcn_diffuse_buffer'] * (data['kpcn_albedo'] + eps) + torch.exp(data['kpcn_specular_buffer']) - 1.0
        inputGrid = torchvision.utils.make_grid(inputFinal)
        writer.add_image('noisy patches e{}'.format(epoch+1), inputGrid)

        outputGrid = torchvision.utils.make_grid(outputFinal)
        writer.add_image('denoised patches e{}'.format(epoch+1), outputGrid)

        cleanGrid = torchvision.utils.make_grid(Y_final)
        writer.add_image('clean patches e{}'.format(epoch+1), cleanGrid)



      batch_idx += 1


    return lossDiff/(4*len(dataloader)), lossSpec/(4*len(dataloader)), lossFinal/(4*len(dataloader)), relL2Final/(4*len(dataloader)), lossAlbedo/(4*len(dataloader)), lossDepth/(4*len(dataloader)), lossNormal/(4*len(dataloader))

def train(mode, device, trainset, validset, eps, L, input_channels, hidden_channels, kernel_size, epochs, learning_rate, loss, do_early_stopping, do_feature_dropout, do_finetune):
  # print('TRAINING WITH VALIDDATASET : {}'.format(validset))
  dataloader = DataLoader(trainset, batch_size=8, num_workers=1, pin_memory=False)
  # print(len(dataloader))

  if validset is not None:
    validDataloader = DataLoader(validset, batch_size=4, num_workers=1, pin_memory=False)

  # instantiate networks
  print(L, input_channels, hidden_channels, kernel_size, mode)
  if 'eca' in mode:
    diffuseNet = make_ecanet(L, input_channels, hidden_channels, kernel_size, mode).to(device)
    specularNet = make_ecanet(L, input_channels, hidden_channels, kernel_size, mode).to(device)
  elif 'cbam' in mode:
    diffuseNet = make_cbamnet(L, input_channels, hidden_channels, kernel_size, mode).to(device)
    specularNet = make_cbamnet(L, input_channels, hidden_channels, kernel_size, mode).to(device)
  elif 'simple_feat' in mode:
    diffuseNet = make_simple_feat_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)
    specularNet = make_simple_feat_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)
  elif 'dense_unet' in mode:
    diffuseNet = make_DenseUnet(input_channels, mode).to(device)
    specularNet = make_DenseUnet(input_channels, mode).to(device)
  elif 'spc' in mode:
    diffuseNet = make_spc_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)
    specularNet = make_spc_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)
  else:
    print(mode)
    diffuseNet = make_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)
    specularNet = make_net(L, input_channels, hidden_channels, kernel_size, mode).to(device)

  encodeNet = encoderUnet().to(device)
  albedoNet, depthNet, normalNet = decoderUnet(mode=0).to(device), decoderUnet(out_channel=1, mode=2).to(device), decoderUnet(mode=1).to(device)

  print(diffuseNet, "CUDA:", next(diffuseNet.parameters()).is_cuda)
  print(specularNet, "CUDA:", next(specularNet.parameters()).is_cuda)
  print(encodeNet, "CUDA:", next(encodeNet.parameters()).is_cuda)
  print(albedoNet, "CUDA:", next(albedoNet.parameters()).is_cuda)

  print('# Parameter for diffuseNet : {}'.format(sum([p.numel() for p in diffuseNet.parameters()])))
  print('# Parameter for specularNet : {}'.format(sum([p.numel() for p in specularNet.parameters()])))
  print('# Parameter for encodeNet : {}'.format(sum([p.numel() for p in encodeNet.parameters()])))
  print('# Parameter for albedoNet, depthNet, normalNet : {}'.format(sum([p.numel() for p in albedoNet.parameters()])))

  
  if loss == 'L1':
    criterion = nn.L1Loss()
  else:
    print('Loss Not Supported')
    return
  print('LEARNING RATE : {}'.format(learning_rate))
  optimizerDiff = optim.Adam(diffuseNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))
  optimizerSpec = optim.Adam(specularNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))
  optimizerEnc = optim.Adam(encodeNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))
  optimizerA = optim.Adam(albedoNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))
  optimizerD = optim.Adam(depthNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))
  optimizerN = optim.Adam(normalNet.parameters(), lr=learning_rate, betas=(0.9, 0.99))

  # checkpointDiff = torch.load('trained_model/multitask/diff_e2.pt')
  # diffuseNet.load_state_dict(checkpointDiff['model_state_dict'])
  # optimizerDiff.load_state_dict(checkpointDiff['optimizer_state_dict'])
  # diffuseNet.load_state_dict(torch.load('trained_model/simple_feat_kpcn_2/diff_e8.pt'))

  # checkpointSpec = torch.load('trained_model/multitask/spec_e2.pt')
  # specularNet.load_state_dict(checkpointSpec['model_state_dict'])
  # optimizerSpec.load_state_dict(checkpointSpec['optimizer_state_dict'])
  # specularNet.load_state_dict(torch.load('trained_model/simple_feat_kpcn_2/spec_e8.pt'))

  # checkpointEncode = torch.load('trained_model/multitask/encode_e2.pt')
  # encodeNet.load_state_dict(checkpointEncode['model_state_dict'])
  # optimizerEnc.load_state_dict(checkpointEncode['optimizer_state_dict'])

  # checkpointAlbedo = torch.load('trained_model/multitask/albedo_e2.pt')
  # albedoNet.load_state_dict(checkpointAlbedo['model_state_dict'])
  # optimizerA.load_state_dict(checkpointAlbedo['optimizer_state_dict'])

  # checkpointDepth = torch.load('trained_model/multitask/depth_e2.pt')
  # depthNet.load_state_dict(checkpointDepth['model_state_dict'])
  # optimizerD.load_state_dict(checkpointDepth['optimizer_state_dict'])

  # checkpointNormal = torch.load('trained_model/multitask/normal_e2.pt')
  # normalNet.load_state_dict(checkpointNormal['model_state_dict'])
  # optimizerN.load_state_dict(checkpointNormal['optimizer_state_dict'])
  
  accuLossDiff, accuLossSpec, accuLossFinal = 0, 0, 0
  accuLossAlbedo, accuLossDepth, accuLossNormal = 0, 0, 0
  
  lDiff = []
  lSpec = []
  lFinal = []
  valLDiff = []
  valLSpec = []
  valLFinal = []

  # writer = SummaryWriter('runs/'+mode+'_2')
  total_epoch = 0
  models = {'diffuse': diffuseNet, 'specular': specularNet, 'encode': encodeNet, 'albedo': albedoNet, 'depth': depthNet, 'normal': normalNet}
  print('Check Initialization')
  initLossDiff, initLossSpec, initLossFinal, relL2LossFinal, initLossAlbedo, initLossDepth, initLossNormal = validation(models, validDataloader, eps, criterion, device, -1, mode)
  print("initLossDiff: {}".format(initLossDiff))
  print("initLossSpec: {}".format(initLossSpec))
  print("initLossFinal: {}".format(initLossFinal))
  print("relL2LossFinal: {}".format(relL2LossFinal))
  print("initLossAlbedo: {}".format(initLossAlbedo))
  print("initLossDepth: {}".format(initLossDepth))
  print("initLossNormal: {}".format(initLossNormal))
  epoch = 0
  if epoch == 0:
    writer.add_scalar('Valid total relL2 loss', relL2LossFinal if relL2LossFinal != float('inf') else 0, epoch)
    writer.add_scalar('Valid total loss', initLossFinal if initLossFinal != float('inf') else 0, epoch)
    writer.add_scalar('Valid diffuse loss', initLossDiff if initLossDiff != float('inf') else 0, epoch)
    writer.add_scalar('Valid specular loss', initLossSpec if initLossSpec != float('inf') else 0, epoch)
    writer.add_scalar('Valid albedo loss', initLossAlbedo if initLossAlbedo != float('inf') else 0, epoch)
    writer.add_scalar('Valid depth loss', initLossDepth if initLossDepth != float('inf') else 0, epoch)
    writer.add_scalar('Valid normal loss', initLossNormal if initLossNormal != float('inf') else 0, epoch)


  import time

  start = time.time()
  print('START')

  for epoch in range(0, epochs):
    print('EPOCH {}'.format(epoch+1))
    diffuseNet.train()
    specularNet.train()
    encodeNet.train()
    albedoNet.train()
    depthNet.train()
    normalNet.train()
    # for i_batch, sample_batched in enumerate(dataloader):
    i_batch = -1
    for sample_batched in tqdm(dataloader, leave=False, ncols=70):
      
      i_batch += 1
      # print(sample_batched.keys())

      # get the inputs
      X_diff = sample_batched['kpcn_diffuse_in'].to(device)
      # X_diff = feature_dropout(X_diff, 1.0, device)

      Y_diff = sample_batched['target_diffuse'].to(device)
      # Y_diff = feature_dropout(Y_diff, 1.0, device)
      # zero the parameter gradients
      optimizerDiff.zero_grad()

      # forward + backward + optimize
      outputDiff = diffuseNet(X_diff)

      # if mode == 'KPCN':
      if 'kpcn' in mode:
        X_input = crop_like(X_diff, outputDiff)
        outputDiff = apply_kernel(outputDiff, X_input, device)

      Y_diff = crop_like(Y_diff, outputDiff)


      # get the inputs
      X_spec = sample_batched['kpcn_specular_in'].to(device)
      Y_spec = sample_batched['target_specular'].to(device)

      # zero the parameter gradients
      optimizerSpec.zero_grad()

      # forward + backward + optimize
      outputSpec = specularNet(X_spec)

      # if mode == 'KPCN':
      if 'kpcn' in mode:
        X_input = crop_like(X_spec, outputSpec)
        outputSpec = apply_kernel(outputSpec, X_input, device)

      Y_spec = crop_like(Y_spec, outputSpec)

      lossDiff = criterion(outputDiff, Y_diff)
      lossSpec = criterion(outputSpec, Y_spec)
      # if epoch >= 0:
      # if not do_finetune:
      #   lossDiff.backward()
      #   optimizerDiff.step()
      #   lossSpec.backward()
      #   optimizerSpec.step()

      # calculate final ground truth error
      # with torch.no_grad():
      # albedo = sample_batched['origAlbedo'].permute(permutation).to(device)
      albedo = sample_batched['kpcn_albedo'].to(device)
      albedo = crop_like(albedo, outputDiff)
      normal = crop_like(sample_batched['kpcn_diffuse_in'][:,10:13,:,:], outputDiff).to(device)
      depth = crop_like(sample_batched['kpcn_diffuse_in'][:,20,:,:], outputDiff).to(device)
      # print('ALBEDO SIZE: {}'.format(sample_batched['kpcn_albedo'].shape))
      outputFinal = outputDiff * (albedo + eps) + torch.exp(outputSpec) - 1.0

      Y_final = sample_batched['target_total'].to(device)
      Y_final = crop_like(Y_final, outputFinal)

      lossFinal = criterion(outputFinal, Y_final)

      optimizerEnc.zero_grad()
      optimizerA.zero_grad()
      optimizerD.zero_grad()
      optimizerN.zero_grad()

			# g-buffer estimation
      enc = encodeNet(outputFinal) # 8 * 3 * 96 * 96
      outputAlbedo, outputDepth, outputNormal = albedoNet(enc[0], enc[1], enc[2], enc[3]), depthNet(enc[0], enc[1], enc[2], enc[3]), normalNet(enc[0], enc[1], enc[2], enc[3])
      # print('ALBEDO SIZE : {}, {}'.format(outputAlbedo.shape, albedo.shape))
      lossAlbedo, lossDepth, lossNormal = criterion(outputAlbedo, albedo), criterion(outputDepth, depth.unsqueeze(1)), criterion(outputNormal, normal)
      # lossAlbedo = criterion(outputAlbedo, albedo)
      # loss_finetune = lossDiff + lossSpec + 0.1 * lossAlbedo + 0.1 * lossDepth + 0.1 * lossNormal
      loss_finetune = lossDiff + lossSpec + lossAlbedo
      loss_finetune.backward()
      optimizerDiff.step()
      optimizerSpec.step()
      optimizerEnc.step()
      optimizerA.step()
      # optimizerD.step()
      # optimizerN.step()

      # if do_finetune:
      #   # print('FINETUNING')

      #   # g-buffer estimation
      #   enc = encodeNet(outputFinal) # 8 * 3 * 96 * 96
      #   outputAlbedo, outputDepth, outputNormal = albedoNet(enc), depthNet(enc), normalNet(enc)
      #   loss_finetune = lossFinal + criterion(outputAlbedo, albedo) + criterion(outputDepth, depth) + criterion(outputNormal, normal)

      #   loss_finetune.backward()
      #   optimizerDiff.step()
      #   optimizerSpec.step()
      accuLossFinal += lossFinal.item()

      accuLossDiff += lossDiff.item()
      accuLossSpec += lossSpec.item()
      accuLossAlbedo += lossAlbedo.item()
      accuLossDepth += lossDepth.item()
      accuLossNormal += lossNormal.item()

      writer.add_scalar('lossFinal',  lossFinal if lossFinal != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
      writer.add_scalar('lossDiffuse', lossDiff if lossDiff != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
      writer.add_scalar('lossSpec', lossSpec if lossSpec != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
      writer.add_scalar('lossAlbedo', lossAlbedo if lossAlbedo != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
      writer.add_scalar('lossDepth', lossDepth if lossDepth != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
      writer.add_scalar('lossNormal', lossNormal if lossNormal != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)


      
    
    accuLossDiff, accuLossSpec, accuLossFinal = accuLossDiff/(8*len(dataloader)), accuLossSpec/(8*len(dataloader)), accuLossFinal/(8*len(dataloader))
    writer.add_scalar('Train total loss', accuLossFinal if accuLossFinal != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
    writer.add_scalar('Train diffuse loss', accuLossDiff if accuLossDiff != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
    writer.add_scalar('Train specular loss', accuLossSpec if accuLossSpec != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
    writer.add_scalar('Train albedo loss', accuLossAlbedo if accuLossAlbedo != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
    writer.add_scalar('Train depth loss', accuLossDepth if accuLossDepth != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)
    writer.add_scalar('Train normal loss', accuLossNormal if accuLossNormal != float('inf') else 1e+35, epoch * len(dataloader) + i_batch)



    if not os.path.exists('trained_model/' + save_dir):
      os.makedirs('trained_model/' + save_dir)
      print('MAKE DIR {}'.format('trained_model/'+save_dir))

    # torch.save(diffuseNet.state_dict(), 'trained_model/'+ save_dir + '/diff_e{}.pt'.format(epoch+1))
    torch.save({
            'epoch': epoch,
            'model_state_dict': diffuseNet.state_dict(),
            'optimizer_state_dict': optimizerDiff.state_dict(),
            }, 'trained_model/'+ save_dir + '/diff_e{}.pt'.format(epoch+1))
    # torch.save(specularNet.state_dict(), 'trained_model/' + save_dir + '/spec_e{}.pt'.format(epoch+1))
    torch.save({
            'epoch': epoch,
            'model_state_dict': specularNet.state_dict(),
            'optimizer_state_dict': optimizerSpec.state_dict(),
            }, 'trained_model/'+ save_dir + '/spec_e{}.pt'.format(epoch+1))
    torch.save({
            'epoch': epoch,
            'model_state_dict': encodeNet.state_dict(),
            'optimizer_state_dict': optimizerEnc.state_dict(),
            }, 'trained_model/'+ save_dir + '/encode_e{}.pt'.format(epoch+1))
    torch.save({
            'epoch': epoch,
            'model_state_dict': albedoNet.state_dict(),
            'optimizer_state_dict': optimizerA.state_dict(),
            }, 'trained_model/'+ save_dir + '/albedo_e{}.pt'.format(epoch+1))
    torch.save({
            'epoch': epoch,
            'model_state_dict': depthNet.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict(),
            }, 'trained_model/'+ save_dir + '/depth_e{}.pt'.format(epoch+1))
    torch.save({
            'epoch': epoch,
            'model_state_dict': normalNet.state_dict(),
            'optimizer_state_dict': optimizerN.state_dict(),
            }, 'trained_model/'+ save_dir + '/normal_e{}.pt'.format(epoch+1))


    # Validation
    models = {'diffuse': diffuseNet, 'specular': specularNet, 'encode': encodeNet, 'albedo': albedoNet, 'depth': depthNet, 'normal': normalNet}
    validLossDiff, validLossSpec, validLossFinal, relL2LossFinal, validLossAlbedo, validLossDepth, validLossNormal = validation(models, validDataloader, eps, criterion, device, epoch, mode)
    writer.add_scalar('Valid total relL2 loss', relL2LossFinal if relL2LossFinal != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
    writer.add_scalar('Valid total loss', validLossFinal if validLossFinal != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
    writer.add_scalar('Valid diffuse loss', validLossDiff if validLossDiff != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
    writer.add_scalar('Valid specular loss', validLossSpec if validLossSpec != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
    writer.add_scalar('Valid albedo loss', validLossAlbedo if validLossAlbedo != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
    writer.add_scalar('Valid depth loss', validLossDepth if validLossDepth != float('inf') else 1e+35, (epoch + 1) * len(dataloader))
    writer.add_scalar('Valid normal loss', validLossNormal if validLossNormal != float('inf') else 1e+35, (epoch + 1) * len(dataloader))

    print("Epoch {}".format(epoch + 1))
    print("TrainLossDiff: {}".format(accuLossDiff))
    print("TrainLossSpec: {}".format(accuLossSpec))
    print("TrainLossFinal: {}".format(accuLossFinal))
    print("ValidrelL2LossDiff: {}".format(relL2LossFinal))
    print("ValidLossDiff: {}".format(validLossDiff))
    print("ValidLossSpec: {}".format(validLossSpec))
    print("ValidLossFinal: {}".format(validLossFinal))
    print("ValidLossAlbedo: {}".format(validLossAlbedo))
    print("ValidLossDepth: {}".format(validLossDepth))
    print("ValidLossNormal: {}".format(validLossNormal))

    lDiff.append(accuLossDiff)
    lSpec.append(accuLossSpec)
    lFinal.append(accuLossFinal)
    valLDiff.append(validLossDiff)
    valLSpec.append(validLossSpec)
    valLFinal.append(validLossFinal)

    print('SAVED {}/diff_e{}, {}/spec_e{}'.format(save_dir, epoch+1, save_dir, epoch+1))

    total_epoch += 1
    if do_early_stopping and len(valLFinal) > 10 and valLFinal[-1] >= valLFinal[-2]:
      print('EARLY STOPPING!')
      break
    
    accuLossDiff = 0
    accuLossSpec = 0
    accuLossFinal = 0

  writer.close()
  print('Finished training in mode, {} with epoch {}'.format(mode, total_epoch))
  print('Took', time.time() - start, 'seconds.')
  models = {'diffuse': diffuseNet, 'specular': specularNet, 'encode': encodeNet, 'albedo': albedoNet, 'depth': depthNet, 'normal': normalNet}
  return models


def main():
  args = parser.parse_args()
  print(args)

  dataset, dataloader = init_data(args)
  print(len(dataset['train']), len(dataloader['train']))
  # trainset, validset = dataloader['train'], dataloader['val']
  trainset, validset = dataset['train'], dataset['val']
  print(trainset, validset)

  input_channels = dataset['train'].dncnn_in_size

  train(
    args.mode, 
    args.device, 
    trainset, 
    validset, 
    eps, 
    args.num_layers, 
    input_channels, 
    args.hidden_channels, 
    args.kernel_size, 
    args.epochs, 
    args.lr, 
    args.loss, 
    args.do_early_stopping, 
    args.do_feature_dropout,
    args.do_finetune)
  


if __name__ == '__main__':
  main()