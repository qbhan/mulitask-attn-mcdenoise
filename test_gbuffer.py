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

# from test_cython import *

# L = 9 # number of convolutional layers
# n_kernels = 100 # number of kernels in each layer
# kernel_size = 5 # size of kernel (square)

# # input_channels = dataset[0]['X_diff'].shape[-1]
# hidden_channels = 100

permutation = [0, 3, 1, 2]
eps = 0.00316

parser = argparse.ArgumentParser(description='Test the model')


parser.add_argument('--device', default='cuda:0')
parser.add_argument('--mode', default='kpcn')
parser.add_argument('--num_layers', default=9, type=int)
parser.add_argument('--input_channels', default=34, type=int)
parser.add_argument('--hidden_channels', default=100, type=int)
parser.add_argument('--kernel_size', default=5, type=int)

parser.add_argument('--diffuse_model')
parser.add_argument('--specular_model')
parser.add_argument('--encoder_model')
parser.add_argument('--albedo_model')
parser.add_argument('--normal_model')
parser.add_argument('--depth_model')


parser.add_argument('--data_dir')
parser.add_argument('--save_dir')

parser.set_defaults(do_vis_feat=False)
parser.add_argument('--do_vis_feat', dest='do_vis_feat', action='store_true')
# parser.set_defaults(do_errormap=False)
# parser.add_argument('--do_errormap', dest='do_errormap', action='store_true')

def unsqueeze_all(d):
  for k, v in d.items():
    d[k] = torch.unsqueeze(v, dim=0)
  return d


def denoise(models, dataloader, device, mode, save_dir, do_vis_feat, debug=False):
  print(len(dataloader))
  diffuseNet = models['diffuse']
  specularNet = models['specular']
  encodeNet = models['encode']
  albedoNet = models['albedo']
  depthNet = models['depth']
  normalNet = models['normal']

  with torch.no_grad():
    criterion = nn.L1Loss()
    relL2 = RelativeMSE()
    lossDiff, lossSpec, lossFinal, relL2Final, lossAlbedo, lossNormal, lossDepth= 0,0,0,0,0,0,0
    image_idx = 0
    input_image = torch.zeros((3, 960, 960)).to(device)
    gt_image = torch.zeros((3, 960, 960)).to(device)
    output_image = torch.zeros((3, 960, 960)).to(device)
    albedo_image = torch.zeros((3, 960, 960)).to(device)
    depth_image = torch.zeros((1, 960, 960)).to(device)
    normal_image = torch.zeros((3, 960, 960)).to(device)

    # Error calculation
    error_map = torch.zeros((3, 960, 960)).to(device)

    x, y = 0, 0
    for data in tqdm(dataloader, leave=False, ncols=70):
      # print(x, y)
      X_diff = data['kpcn_diffuse_in'].to(device)
      Y_diff = data['target_diffuse'].to(device)

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

      Y_final = data['target_total'].to(device)
      Y_final = crop_like(Y_final, outputFinal)
      # print(lossFinal, relL2Final)
      lossFinal += criterion(outputFinal, Y_final).item()
      relL2Final += relL2(outputFinal, Y_final).item()

    
			# g-buffer estimation
      normal = crop_like(data['kpcn_diffuse_in'][:,10:13,:,:], outputDiff).to(device)
      depth = crop_like(data['kpcn_diffuse_in'][:,20,:,:], outputDiff).to(device)

      enc = encodeNet(outputFinal)
      outputAlbedo, outputDepth, outputNormal = albedoNet(enc[0], enc[1], enc[2], enc[3]), depthNet(enc[0], enc[1], enc[2], enc[3]), normalNet(enc[0], enc[1], enc[2], enc[3])
      lossAlbedo +=  criterion(outputAlbedo, albedo).item()
      lossDepth += criterion(outputDepth, depth.unsqueeze(1)).item()
      lossNormal += criterion(outputNormal, normal).item()

      # visualize
      inputFinal = data['kpcn_diffuse_buffer'] * (data['kpcn_albedo'] + eps) + torch.exp(data['kpcn_specular_buffer']) - 1.0


      # print(np.shape(inputFinal))
      # print(np.shape(outputFinal))
      # print(np.shape(Y_final))
      input_image[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(inputFinal[0, :, 32:96, 32:96])
      output_image[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(outputFinal[0, :, 16:80, 16:80])
      gt_image[:, x*64:x*64+64, y*64:y*64+64] = ToneMapTest(Y_final[0, :, 16:80, 16:80])
      error_map[:, x*64:x*64+64, y*64:y*64+64] = torch.abs(gt_image[:, x*64:x*64+64, y*64:y*64+64] - output_image[:, x*64:x*64+64, y*64:y*64+64])
      # print(albedo_image[:, x*64:x*64+64, y*64:y*64+64].shape, outputAlbedo.shape)
      albedo_image[:, x*64:x*64+64, y*64:y*64+64] = outputAlbedo[0, :, 16:80, 16:80]
      depth_image[:, x*64:x*64+64, y*64:y*64+64] = outputDepth[0, :, 16:80, 16:80]
      normal_image[:, x*64:x*64+64, y*64:y*64+64] = outputNormal[0, :, 16:80, 16:80]

      if 'simple_feat' in mode:
        if not os.path.exists(save_dir + '/test{}/attns'.format(image_idx)):
            os.makedirs(save_dir + '/test{}/attns'.format(image_idx))
        foutDiff = open(save_dir + '/test{}/attns/patch_{}_diff_attn.csv'.format(image_idx, 15*x+y), 'w')
        foutSpec = open(save_dir + '/test{}/attns/patch_{}_spec_attn.csv'.format(image_idx, 15*x+y), 'w')
        for fi in range(0, 9):
          pass
          fDiff = open('acts/Act_diff_{}.csv'.format(fi))
          fDiff.__next__()
          fSpec = open('acts/Act_spec_{}.csv'.format(fi))
          fSpec.__next__()
          for lineDiff in fDiff:
            # print(line)
            foutDiff.write(str(fi)+ lineDiff[1:])
          for lineSpec in fSpec:
            foutSpec.write(str(fi) + lineSpec[1:])
          fDiff.close()
          fSpec.close()
        foutDiff.close()
        foutSpec.close()

      elif 'spc' in mode:
        if not os.path.exists(save_dir + '/test{}/attns'.format(image_idx)):
            os.makedirs(save_dir + '/test{}/attns'.format(image_idx))
        fout = open(save_dir + '/test{}/attns/patch_{}_attn.csv'.format(image_idx, 15*x+y), 'w')
        for fi in range(9):
          for fj in range(21):
            f = open('acts/spc/Act_{}_{}.csv'.format(fi, fj))
            f.__next__()
            for line in f:
              # print(line)
              fout.write(str(fi)+ line[1:])
            f.close()
        fout.close()


      
      y += 1
      if x < 15 and y>=15:
        x += 1
        y = 0

      if x >= 15:
        if not os.path.exists(save_dir + '/test{}'.format(image_idx)):
          os.makedirs(save_dir + '/test{}'.format(image_idx))
        if not os.path.exists(save_dir + '/test{}/features'.format(image_idx)):
          os.makedirs(save_dir + '/test{}/features'.format(image_idx))
        # if not os.path.exists(save_dir + '/test{}/attns'.format(image_idx)):
        #   os.makedirs(save_dir + '/test{}/attns'.format(image_idx))

        save_image(input_image, save_dir + '/test{}/noisy.png'.format(image_idx))
        save_image(output_image, save_dir + '/test{}/denoise.png'.format(image_idx))
        save_image(gt_image, save_dir + '/test{}/clean.png'.format(image_idx))
        save_image(error_map, save_dir + '/test{}/error_map.png'.format(image_idx))
        save_image(albedo_image, save_dir + '/test{}/albedo.png'.format(image_idx))
        save_image(depth_image, save_dir + '/test{}/depth.png'.format(image_idx))
        save_image(normal_image, save_dir + '/test{}/normal.png'.format(image_idx))
          
        
        # print('SAVED IMAGES')
        x, y = 0, 0
        image_idx += 1


  return lossDiff/len(dataloader), lossSpec/len(dataloader), lossFinal/len(dataloader), relL2Final/len(dataloader), lossAlbedo/len(dataloader), lossDepth/len(dataloader), lossNormal/len(dataloader)

def test_model(models, device, data_dir, mode, args):
  pass
  # diffuseNet.to(device)
  # specularNet.to(device)
  dataset = DenoiseDataset(data_dir, 8, 'kpcn', 'test', 1, 'recon',
        use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, pnet_out_size=3)
  dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=False
    )
  _, _, totalL1, totalrelL2, totalAlbedo, totalDepth, totalNormal = denoise(models, dataloader, device, mode, args.save_dir, args.do_vis_feat)
  print('TEST L1 LOSS is {}'.format(totalL1))
  print('TEST L2 LOSS is {}'.format(totalrelL2))
  print('TEST ALBEDO LOSS is {}'.format(totalAlbedo))
  print('TEST DEPTH LOSS is {}'.format(totalDepth))
  print('TEST NORMAL LOSS is {}'.format(totalNormal))


def main():
  args = parser.parse_args()
  print(args)

  if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

  if 'simple_feat' in args.mode:
    diffuseNet = make_simple_feat_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode).to(args.device)
    specularNet = make_simple_feat_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode, branch='spec').to(args.device)
  elif 'spc' in args.mode:
    diffuseNet = make_spc_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode).to(args.device)
    specularNet = make_spc_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode).to(args.device)
  else:
    diffuseNet = make_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode).to(args.device)
    specularNet = make_net(args.num_layers, args.input_channels, args.hidden_channels, args.kernel_size, args.mode).to(args.device)

  encodeNet = encoderUnet().to(args.device)
  albedoNet, depthNet, normalNet = decoderUnet(mode=0).to(args.device), decoderUnet(out_channel=1, mode=2).to(args.device), decoderUnet(mode=1).to(args.device)

  # print(diffuseNet, "CUDA:", next(diffuseNet.parameters()).is_cuda)
  # print(torch.load(args.diffuse_model).keys)
  checkpointDiff = torch.load(args.diffuse_model)
  checkpointSpec = torch.load(args.specular_model)
  checkpointEnc = torch.load(args.encoder_model)
  checkpointA = torch.load(args.albedo_model)
  checkpointN = torch.load(args.normal_model)
  checkpointD = torch.load(args.depth_model)
  diffuseNet.load_state_dict(checkpointDiff['model_state_dict'])
  specularNet.load_state_dict(checkpointSpec['model_state_dict'])
  encodeNet.load_state_dict(checkpointEnc['model_state_dict'])
  albedoNet.load_state_dict(checkpointA['model_state_dict'])
  normalNet.load_state_dict(checkpointN['model_state_dict'])
  depthNet.load_state_dict(checkpointD['model_state_dict'])
  models = {'diffuse': diffuseNet, 'specular': specularNet, 'encode': encodeNet, 'albedo': albedoNet, 'depth': depthNet, 'normal': normalNet}
  test_model(models, args.device, args.data_dir, args.mode, args)



if __name__ == '__main__':
  main()