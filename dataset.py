import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

import os
import math
import time
import random
import sys

import visdom
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, sobel
from scipy.ndimage.filters import laplace
import matplotlib.pyplot as plt

from utils import ToneMap, LinearToSrgb

random.seed("Kyubeom Han")
# random.seed("Inyoung Cho")
# random.seed("Inyoung Cho, Yuchi Huo, Sungeui Yoon @ KAIST")


def gradient_importance_map(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_x_0 = sobel(gaussian_filter(img[:,:,0], 31), axis=0, mode='nearest')
        img_y_0 = sobel(gaussian_filter(img[:,:,0], 31), axis=1, mode='nearest')
        img_x_1 = sobel(gaussian_filter(img[:,:,1], 31), axis=0, mode='nearest')
        img_y_1 = sobel(gaussian_filter(img[:,:,1], 31), axis=1, mode='nearest')
        img_x_2 = sobel(gaussian_filter(img[:,:,2], 31), axis=0, mode='nearest')
        img_y_2 = sobel(gaussian_filter(img[:,:,2], 31), axis=1, mode='nearest')
        
        img_ = np.sqrt(img_x_0 * img_x_0 + img_y_0 * img_y_0 + 
                        img_x_1 * img_x_1 + img_y_1 * img_y_1 + 
                        img_x_2 * img_x_2 + img_y_2 * img_y_2)
    elif len(img.shape) == 2 or img.shape[2] == 1:
        img_x_0 = sobel(gaussian_filter(img[:,:], 31), axis=0, mode='nearest')
        img_y_0 = sobel(gaussian_filter(img[:,:], 31), axis=1, mode='nearest')
        img_ = np.sqrt(img_x_0 * img_x_0 + img_y_0 * img_y_0)
    else:
        assert False, 'Input image should be either rgb or gray scale format.'
    
    return (img_ - np.min(img_)) / (np.max(img_) - np.min(img_) + 1E-5)

    
def get_valid_path(path):
    if not os.path.isfile(path):
        if os.sep + 'ssd' in path:
            idx1 = path.rfind(os.sep + 'ssd')
        elif os.sep + 'hdd' in path:
            idx1 = path.rfind(os.sep + 'hdd')
        else:
            raise FileNotFoundError(path)
        idx2 = path.find(os.sep, idx1+1)

        for disk in ['ssd1', 'ssd2', 'ssd3', 'hdd1']:
            tmp_p = path.replace(path[idx1:idx2], os.sep + disk)
            if os.path.isfile(tmp_p):
                return tmp_p

        raise FileNotFoundError(path)
    else:
        return path


class DenoiseDataset(Dataset):

    """Mode to preprocess features to the format expected by [Gharbi2019]"""
    SBMC = "sbmc"
    """Mode to preprocess features to the format expected by [Munkberg2020]"""
    LBMC = "lbmc"
    """Mode to preprocess features to the format expected by [Bako2017]"""
    KPCN = "kpcn"
    """Mode to preprocess features to the format expected by [Xu2019]"""
    AMCD = "amcd"

    MAX_DEPTH = 5

    PATCH_SIZE = 128 #128

    """LLPM input set
        'paths':
            "path_weight",
            "radiance_wo_weight_r", "radiance_wo_weight_g", "radiance_wo_weight_b",
            "light_intensity_r", "light_intensity_g", "light_intensity_b",
            "throughputs_r", "throughputs_g", "throughputs_b", * (self.MAX_DEPTH+1)
            "bounce_types", * (self.MAX_DEPTH+1)
            "roughnesses", * (self.MAX_DEPTH+1)
    """

    """SBMC input set
        'target_image':
            "gt_total_r", "gt_total_g", "gt_total_b", # linear
        'radiance':
            "total_r", "total_g", "total_b", # linear
        'features':
            "total_r", "total_g", "total_b", # log
            "specular_r", "specular_g", "specular_b", # log
            "subpixel_x", "subpixel_x",
            "albedo_first_r", "albedo_first_g", "albedo_first_b",
            "albedo_r", "albedo_g", "albedo_b",
            "normal_first_x", "normal_first_y", "normal_first_z",
            "normal_x", "normal_y", "normal_z",
            "depth_first",
            "depth",
            "visibility",
            "hasHit", # 24
            "prob_b1", "prob_b2", "prob_l1", "prob_l2", * (self.MAX_DEPTH+1)
            "l_dir_theta", "l_dir_phi", * (self.MAX_DEPTH+1)
            "reflection", * (self.MAX_DEPTH+1)
            "transmission", * (self.MAX_DEPTH+1)
            "diffuse", * (self.MAX_DEPTH+1)
            "glossy", * (self.MAX_DEPTH+1)
            "specular", * (self.MAX_DEPTH+1)
    """

    """LBMC input set
        'target_image':
            "gt_total_r", "gt_total_g", "gt_total_b", # linear
        'radiance':
            "total_r", "total_g", "total_b", # linear
        'features':
            "total_r", "total_g", "total_b", # log
            "specular_r", "specular_g", "specular_b", # log
            "subpixel_x", "subpixel_x",
            "albedo_first_r", "albedo_first_g", "albedo_first_b",
            "albedo_r", "albedo_g", "albedo_b",
            "normal_first_x", "normal_first_y", "normal_first_z",
            "normal_x", "normal_y", "normal_z",
            "depth_first",
            "depth",
            "visibility",
            "hasHit", # 24
    """

    """KPCN input set
        'target_total':
            gt_r, gt_g, gt_b, # linear
        'target_diffuse':
            gt_diffuse_r, gt_diffuse_g, gt_diffuse_b, # albedo factored
        'target_specular':
            gt_specular_r, gt_specular_g, gt_specular_b, # log 
        'kpcn_diffuse_in':
            diffuse_r, diffuse_g, diffuse_b, diffuse_v, # albedo factored
            diffuse_r_dx, diffuse_g_dx, diffuse_b_dx, diffuse_r_dy, diffuse_g_dy, diffuse_b_dy, 
            normals_x, normals_y, normals_z, normals_v, 
            normals_x_dx, normals_y_dx, normals_z_dx, normals_x_dy, normals_y_dy, normals_z_dy, 
            depth, depth_v, 
            depth_dx, depth_dy,
            albedo_r, albedo_g, albedo_b, albedo_v, 
            albedo_r_dx, albedo_g_dx, albedo_b_dx, albedo_r_dy, albedo_g_dy, albedo_b_dy, 
        'kpcn_specular_in':
            specular_r, specular_g, specular_b, specular_v, 
            specular_r_dx, specular_g_dx, specular_b_dx, specular_r_dy, specular_g_dy, specular_b_dy, # log
            normals_x, normals_y, normals_z, normals_v, 
            normals_x_dx, normals_y_dx, normals_z_dx, normals_x_dy, normals_y_dy, normals_z_dy, 
            depth, depth_v, 
            depth_dx, depth_dy,
            albedo_r, albedo_g, albedo_b, albedo_v, 
            albedo_r_dx, albedo_g_dx, albedo_b_dx, albedo_r_dy, albedo_g_dy, albedo_b_dy, 
        'kpcn_diffuse_buffer':
            diffuse_r, diffuse_g, diffuse_b, # albedo factored
        'kpcn_specular_buffer':
            specular_r, specular_g, specular_b, # log
        'kpcn_albedo':
            albedo_r, albedo_g, albedo_b,
    """

    """AMCD input set
        'ref': # diffuse_ref, specular_ref
            gt_diffuse_r, gt_diffuse_g, gt_diffuse_b, # albedo factored
            gt_specular_r, gt_specular_g, gt_specular_b, # log # for diff/spec simultaneous training
        'in': # diffuse_in
            diffuse_r, diffuse_g, diffuse_b, # albedo factored
            specular_r, specular_g, specular_b, # log # for diff/spec simultaneous training
        'seg': 
            normals_x, normals_y, normals_z,
            depth,
            albedo_r, albedo_g, albedo_b,
        #'NOISY_path':
        #'GT_path':
        'x_offset': 128,
        'y_offset': 128,
    """

    def __init__(self, gt_base_dir, spp, base_model='kpcn', mode='train', batch_size=8, sampling='random', use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, pnet_out_size=3):
        if base_model not in [self.SBMC, self.KPCN, self.AMCD, self.LBMC]:
            raise RuntimeError("Unknown baseline model %s" % base_model)
        
        if mode not in ['train', 'val', 'test']:
            raise RuntimeError("Unknown training mode %s" % mode)
        
        if sampling not in ['random', 'grid', 'recon']:
            raise RuntimeError("Unknown training mode %s" % mode)

        if base_model == self.LBMC:
            base_model = self.SBMC
            use_sbmc_buf = False
            use_g_buf = True
        
        # Basic flags
        self.gt_dir = os.path.join(gt_base_dir, mode, 'gt')
        # print(self.gt_dir)
        ssd1_fn = next(os.walk(self.gt_dir))[2]
        ssd1_gt_base = next(os.walk(self.gt_dir))[0]
        #if mode != 'test':
        #    ssd2_fn = next(os.walk(self.gt_dir.replace('ssd1', 'ssd2')))[2]
        #    ssd2_gt_base = next(os.walk(self.gt_dir.replace('ssd1', 'ssd2')))[0]
        self.gt_files = [os.path.join(ssd1_gt_base, s) for s in ssd1_fn] 
        #self.gt_files = random.sample(self.gt_files, len(self.gt_files)//8)
        #if mode != 'test':
        #    self.gt_files += [os.path.join(ssd2_gt_base, s) for s in ssd2_fn]
        self.spp = spp
        self.batch_size = batch_size
        self.mode = mode
        self.sampling = sampling

        # Flags to select which features to load from the disk
        self.base_model = base_model
        self.use_g_buf = use_g_buf
        self.use_sbmc_buf = use_sbmc_buf
        self.use_llpm_buf = use_llpm_buf

        if self.base_model != self.SBMC:
            self.use_sbmc_buf = False
        
        # Model input channel size
        self.pnet_in_size = 0
        if use_llpm_buf:
            self.pnet_in_size += 36
        
        self.pnet_out_size = pnet_out_size
        
        self.dncnn_in_size = 0
        if base_model == self.SBMC:
            self.dncnn_in_size = 3
            if use_g_buf:
                self.dncnn_in_size += 21
            if use_sbmc_buf:
                self.dncnn_in_size += 66
        elif base_model == self.KPCN:
            self.dncnn_in_size = 34
        elif base_model == self.AMCD:
            self.dncnn_in_size = 10

        if use_llpm_buf:
            self.dncnn_in_size += pnet_out_size + 2 # path weight, p-buffer, variance

        # TODO(cho): OptaGen?????? ????????? ????????? ????????? metadata??? ?????? ????????? ??????.
        # Raw feature ranges
        self.idx_gt = {
            'radiance_r': [0,3],
            'diffuse_r': [3,6],
            'albedo_r': [6,9]
        }
        self.idx_nsy = {
            'radiance': [2,5],
            'diffuse': [5,8],
        }
        self.idx_g = {
            'subpixel': [0,2],
            'albedo_at_first': [8,11], # at the first geometric bounce
            'albedo': [11,14], # at the first non-specular bounce
            'normal_at_first': [14,17],
            'normal': [17,20],
            'depth_at_first': [20,21],
            'depth': [21,22],
            'visibility': [22,23],
            'hasHit': [23,24],
            'albedo_at_diff': [24+(self.MAX_DEPTH+1)*7,  # at the first diffuse bounce
                               27+(self.MAX_DEPTH+1)*7],
            'normal_at_diff': [27+(self.MAX_DEPTH+1)*7,
                               30+(self.MAX_DEPTH+1)*7],
            'depth_at_diff': [30+(self.MAX_DEPTH+1)*7,
                               31+(self.MAX_DEPTH+1)*7]
        }
        self.idx_sbmc = {
            'probabilities': [24,24+(self.MAX_DEPTH+1)*4],
            'light_directions': [24+(self.MAX_DEPTH+1)*4,
                                 24+(self.MAX_DEPTH+1)*6],
            'bounce_types': [24+(self.MAX_DEPTH+1)*6,
                             24+(self.MAX_DEPTH+1)*7]
        }
        self.idx_llpm = {
            'path_weight': [31+(self.MAX_DEPTH+1)*7,
                            32+(self.MAX_DEPTH+1)*7],
            'radiance_wo_weight': [32+(self.MAX_DEPTH+1)*7,
                                   35+(self.MAX_DEPTH+1)*7],
            'light_intensity': [35+(self.MAX_DEPTH+1)*7,
                                38+(self.MAX_DEPTH+1)*7],
            'throughputs': [38+(self.MAX_DEPTH+1)*7,
                            38+(self.MAX_DEPTH+1)*10],
            'roughnesses': [38+(self.MAX_DEPTH+1)*10,
                            38+(self.MAX_DEPTH+1)*11]
        }

        # Set random seeds
        random.seed("Kyubeom Han")
        # random.seed("Inyoung Cho")
        # random.seed("Inyoung Cho, Yuchi Huo, Sungeui Yoon @ KAIST")
        random.shuffle(self.gt_files)

        # Constants for patch importance sampling
        if sampling == 'random':
            self.patches_per_image = (256 // batch_size) * batch_size
        elif sampling == 'grid':
            self.patches_per_image = 100
        elif sampling == 'recon':
            self.patches_per_image = 15 * 15 # 19 * 19 for training, 15*15 for test
        else:
            raise RuntimeError("Unknown training mode %s" % mode)
        self.samples = None
    
    def __len__(self):
        # print(len(self.gt_files) * self.patches_per_image)
        return (len(self.gt_files) // 1) * self.patches_per_image
        # print(self.patches_per_image)
        # return self.patches_per_image
        # if self.mode == 'train'
        # return 256

# Preprocessing
    def _gradients(self, buf):
        """Compute the xy derivatives of the input buffer. This helper is used in the _preprocess_<base_model>(...) functions
        Args:
            buf(np.array)[h, w, c]: input image-like tensor.
        Returns:
            (np.array)[h, w, 2*c]: horizontal and vertical gradients of buf.
        """
        dx = buf[:, 1:, ...] - buf[:, :-1, ...]
        dy = buf[1:, ...] - buf[:-1, ...]
        dx = np.pad(dx, [[0, 0], [1, 0], [0, 0]], mode="constant") # zero padding to the left
        dy = np.pad(dy, [[1, 0], [0, 0], [0, 0]], mode='constant')  # zero padding to the up
        return np.concatenate([dx, dy], 2)

    def _preprocess_llpm(self, sample):
        """
        Args:
            _in(numpy.array): raw samples.
        Returns:
            llpm_buffer(numpy.array) which consists of:
                "path_weight",
                "radiance_wo_weight_r", "radiance_wo_weight_g", "radiance_wo_weight_b",
                "light_intensity_r", "light_intensity_g", "light_intensity_b",
                "throughputs_r", "throughputs_g", "throughputs_b", * (self.MAX_DEPTH+1)
                "bounce_types", * (self.MAX_DEPTH+1)
                "roughnesses", * (self.MAX_DEPTH+1)
        """
        feats_list = []

        idx_s, idx_e = self.idx_llpm['path_weight']
        path_weight = sample[..., idx_s:idx_e]
        path_weight = np.log(path_weight + 1e-6) / 90.0

        idx_s, idx_e = self.idx_llpm['radiance_wo_weight']
        radiance_wo_weight = sample[..., idx_s:idx_e]
        radiance_wo_weight = np.log(radiance_wo_weight + 1e-6) / 30.0

        # light_intensity in [0, 1e4]
        idx_s, idx_e = self.idx_llpm['light_intensity']
        light_intensity = sample[..., idx_s:idx_e]
        light_intensity = np.log(light_intensity + 1e-8) / 10.0 

        idx_s, idx_e = self.idx_llpm['throughputs']
        throughputs = sample[..., idx_s:idx_e]
        throughputs = np.log(throughputs + 1e-6) / 30.0

        idx_s, idx_e = self.idx_sbmc['bounce_types']
        bounce_types = sample[..., idx_s:idx_e] / 19.0

        # for visual linearity, we sample roughnesses in U[0,1]^2 in our renderer
        # [Burley2012]
        idx_s, idx_e = self.idx_llpm['roughnesses']
        roughnesses = sample[..., idx_s:idx_e]
        roughnesses = np.sqrt(roughnesses)

        feats_list = [
            path_weight, radiance_wo_weight, 
            light_intensity, throughputs, bounce_types,
            roughnesses
        ]

        """ Feature visualization
        for im in feats_list:
            im = im.mean(2)
            if im.shape[2] % 6 == 0:
                im = im[...,:im.shape[2] // 6]
            if im.shape[2] == 1:
                im = im[...,0]
            plt.imshow(im, vmin=im.min(), vmax=im.max())
            plt.show()
        """
        
        llpm_buffer = np.concatenate(feats_list, axis=3)
        return llpm_buffer
    
    def _preprocess_sbmc(self, sample):
        """
        Args:
            _in(numpy.array): raw samples.
        Returns:
            sbmc_s_buffer(numpy.array) which consists of:
                "total_r", "total_g", "total_b", # linear
                "total_r", "total_g", "total_b", # log
                "specular_r", "specular_g", "specular_b", # log
                "subpixel",
                "normal_first_x", "normal_first_y", "normal_first_z",
                "normal_x", "normal_y", "normal_z",
                "depth_first",
                "depth",
                "visibility",
                "hasHit",
                "albedo_first_r", "albedo_first_g", "albedo_first_b",
                "albedo_r", "albedo_g", "albedo_b",
            sbmc_p_buffer(numpy.array) which consists of:
                "prob_b1", "prob_b2", "prob_l1", "prob_l2", * (self.MAX_DEPTH+1)
                "l_dir_theta", "l_dir_phi", * (self.MAX_DEPTH+1)
                "reflection", * (self.MAX_DEPTH+1)
                "transmission", * (self.MAX_DEPTH+1)
                "diffuse", * (self.MAX_DEPTH+1)
                "glossy", * (self.MAX_DEPTH+1)
                "specular", * (self.MAX_DEPTH+1)
        """
        s_list = [] # samples
        p_list = [] # paths

        # Total radiance
        idx_s, idx_e = self.idx_nsy['radiance']
        total = sample[..., idx_s:idx_e]
        total = np.maximum(total, 0)

        # Diffuse
        idx_s, idx_e = self.idx_nsy['diffuse']
        diffuse = sample[..., idx_s:idx_e]
        diffuse = np.maximum(diffuse, 0)

        # Specular
        specular = np.maximum(total - diffuse, 0)
        specular = np.log(1 + specular) / 10.0

        # Subpixel coordinates
        idx_s, idx_e = self.idx_g['subpixel']
        subpixel = sample[..., idx_s:idx_e]

        # G-buffer of SBMC
        idx_s, _ = self.idx_g['albedo_at_first']
        _, idx_e = self.idx_g['hasHit']
        g_buffer = sample[..., idx_s:idx_e]

        # BRDF and light sampling probabilities
        idx_s, idx_e = self.idx_sbmc['probabilities']
        probabilities = sample[..., idx_s:idx_e]
        probabilities = np.log(np.maximum(probabilities, 0) + 1e-5) / 30.0

        # Path directions (latitude, longitude) from camera coordinates
        idx_s, idx_e = self.idx_sbmc['light_directions']
        light_directions = sample[..., idx_s:idx_e]
        light_directions = np.clip(light_directions, -1.0, 1.0)

        # Light-material interaction tags
        idx_s, idx_e = self.idx_sbmc['bounce_types']
        bounce_types = sample[..., idx_s:idx_e].astype(np.int16)
        is_reflection = np.bitwise_and(bounce_types, 1).astype(np.bool).astype(np.float32)
        is_transmission = np.bitwise_and(
            bounce_types, 1 << 1).astype(np.bool).astype(np.float32)
        is_diffuse = np.bitwise_and(
            bounce_types, 1 << 2).astype(np.bool).astype(np.float32)
        is_glossy = np.bitwise_and(
            bounce_types, 1 << 3).astype(np.bool).astype(np.float32)
        is_specular = np.bitwise_and(
            bounce_types, 1 << 4).astype(np.bool).astype(np.float32)

        s_list = [
            total,
            np.log(1 + total) / 10.0,
            specular,
            subpixel,
            g_buffer
        ]

        p_list = [
            probabilities,
            light_directions,
            is_reflection, is_transmission, is_diffuse, is_glossy, is_specular
        ]

        """ Feature visualization
        for im in s_list:
            im = im.mean(2)
            if im.shape[2] % 6 == 0:
                im = im[...,:im.shape[2] // 6]
            if im.shape[2] == 1:
                im = im[...,0]
            elif im.shape[2] == 2:
                im = im[...,0]
            elif im.shape[2] == 16:
                im = im[...,:3]
            plt.imshow(im, vmin=im.min(), vmax=im.max())
            plt.show()
        """

        """ Feature visualization
        for im in p_list:
            im = im.mean(2)
            if im.shape[2] % 6 == 0:
                im = im[...,:im.shape[2] // 6]
            if im.shape[2] == 1:
                im = im[...,0]
            elif im.shape[2] == 2:
                im = im[...,0]
            plt.imshow(im, vmin=im.min(), vmax=im.max())
            plt.show()
        """

        sbmc_s_buffer = np.concatenate(s_list, axis=3)
        sbmc_p_buffer = np.concatenate(p_list, axis=3)

        return sbmc_s_buffer, sbmc_p_buffer

    def _preprocess_kpcn(self, sample):
        """
        Args:
            _in(numpy.array): raw samples.
        Returns:
            kpcn_buffer(numpy.array) which consists of:
                diffuse_r, diffuse_g, diffuse_b, diffuse_v, # albedo factored
                diffuse_r_dx, diffuse_g_dx, diffuse_b_dx, diffuse_r_dy, diffuse_g_dy, diffuse_b_dy,
                specular_r, specular_g, specular_b, specular_v,
                specular_r_dx, specular_g_dx, specular_b_dx, specular_r_dy, specular_g_dy, specular_b_dy, # log
                normals_x, normals_y, normals_z, normals_v, 
                normals_x_dx, normals_y_dx, normals_z_dx, normals_x_dy, normals_y_dy, normals_z_dy, 
                depth, depth_v, 
                depth_dx, depth_dy,
                albedo_r, albedo_g, albedo_b, albedo_v, 
                albedo_r_dx, albedo_g_dx, albedo_b_dx, albedo_r_dy, albedo_g_dy, albedo_b_dy, 
        """
        spp = sample.shape[2]
        eps = 0.00316

        feats_list = []

        # Normal
        idx_s, idx_e = self.idx_g['normal_at_diff']
        normal = sample[..., idx_s:idx_e].mean(2)
        normal_v = sample[..., idx_s:idx_e].var(2).mean(2, keepdims=True) / spp

        # Depth
        idx_s, idx_e = self.idx_g['depth_at_diff']
        depth = sample[..., idx_s:idx_e].mean(2)
        depth_v = sample[..., idx_s:idx_e].var(2)
        max_depth = depth.max()
        if max_depth > 0:
            depth /= max_depth
            depth_v /= max_depth*max_depth*spp
        depth = np.clip(depth, 0, 1)

        # Albedo
        idx_s, idx_e = self.idx_g['albedo_at_diff']
        albedo = sample[..., idx_s:idx_e].mean(2)
        albedo_v = sample[..., idx_s:idx_e].var(2).mean(2, keepdims=True) / spp
        albedo_sqr = ((albedo + eps)*(albedo + eps)).mean(2, keepdims=True)

        # Diffuse
        idx_s, idx_e = self.idx_nsy['diffuse']
        diff_sample = sample[..., idx_s:idx_e]
        diffuse = np.maximum(diff_sample, 0).mean(2)
        diffuse_v = np.maximum(diff_sample, 0).var(2).mean(2, keepdims=True) / spp
        
        # Specular
        idx_s, idx_e = self.idx_nsy['radiance']
        tot_sample = sample[..., idx_s:idx_e]
        spec_sample = np.maximum(tot_sample, 0) - np.maximum(diff_sample, 0)
        specular = np.maximum(spec_sample, 0).mean(2)
        specular_v = np.maximum(spec_sample, 0).var(2).mean(2, keepdims=True) / spp
        specular_sqr = ((1 + specular)*(1 + specular)).mean(2, keepdims=True) # bug on Gharbi et. al.
        
        # Diffuse: albedo factorization
        diffuse /= albedo + eps
        diffuse_v /= albedo_sqr

        # Specular: log transformation
        specular = np.log(1 + specular)
        specular_v /= specular_sqr # bug on Gharbi et. al.

        # Gradients
        diffuse_g = self._gradients(diffuse)
        specular_g = self._gradients(specular)
        normal_g = self._gradients(normal)
        depth_g = self._gradients(depth)
        albedo_g = self._gradients(albedo)

        feats_list = [
            diffuse, diffuse_v, diffuse_g,
            specular, specular_v, specular_g,
            normal, normal_v, normal_g,
            depth, depth_v, depth_g,
            albedo, albedo_v, albedo_g,
        ]

        """ Feature visualization
        for im in feats_list:
            if im.shape[2] == 1:
                im = im[...,0]
            elif im.shape[2] == 6:
                im = im[...,:3] + im[...,3:]
            elif im.shape[2] == 2:
                im = im[...,0] + im[...,1]
            plt.imshow(im)
            plt.show()
        """
        
        kpcn_buffer = np.concatenate(feats_list, axis=2)

        return kpcn_buffer

    def _offline_preprocess(self, llpm=False, sbmc=False, kpcn=True, overwrite=False, remove_original=False):
        """Preprocess all buffers of indicated models and save them in an offline fashion.
        Args:
            llpm(boolean): process the LLPM [Cho21] buffer.
            sbmc(boolean): process the SBMC [Gharbi19] buffer.
            kpcn(boolean): process the KPCN [Bako17] and AMCD [Xu19] buffer.
            overwrite(boolean): skip processing for existing processed buffer if this sets to False.
            remove_original(boolean): remove vanilla numpy arrays after preprocessing if this sets to True.
        Returns:
            None
        """
        from pathlib import Path

        for gt_fn in tqdm(self.gt_files, leave=False, ncols=100):
            # print(gt_fn)
            # No need to open any .npy file
            in_fn = gt_fn.replace(os.sep + 'gt' + os.sep, os.sep + 'input' + os.sep)
            llpm_fn = in_fn[:in_fn.rfind('.')] + '_llpm' + in_fn[in_fn.rfind('.'):]
            sbmc_s_fn = in_fn[:in_fn.rfind('.')] + '_sbmc_s' + in_fn[in_fn.rfind('.'):]
            sbmc_p_fn = in_fn[:in_fn.rfind('.')] + '_sbmc_p' + in_fn[in_fn.rfind('.'):]
            kpcn_fn = in_fn[:in_fn.rfind('.')] +  '_kpcn_' + str(self.spp) + in_fn[in_fn.rfind('.'):]
            prob_fn = in_fn[:in_fn.rfind('.')] + '_prob_imp' + in_fn[in_fn.rfind('.'):]

            if (not overwrite and 
                (not llpm or os.path.isfile(llpm_fn)) and
                (not sbmc or os.path.isfile(sbmc_s_fn)) and
                (not sbmc or os.path.isfile(sbmc_p_fn)) and
                (not kpcn or os.path.isfile(kpcn_fn)) and
                os.path.isfile(prob_fn)):
                continue

            # Load input sample radiances
            sample = np.load(in_fn, mmap_mode='r')[:,:,:self.spp,:].astype(np.float32)

            assert sample.shape[-1] == 104, 'input numpy file is not produced by OptaGen'

            # NaN handling
            sample = np.where(np.isfinite(sample), sample, 1.0e+38)
            sample = np.where(sample < 1.0e+38, sample, 1.0e+38)

            # Preprocess and save
            if (llpm):
                if (not os.path.isfile(llpm_fn) or overwrite):
                    llpm_buffer = self._preprocess_llpm(sample)
                    np.save(llpm_fn, llpm_buffer)

                for i in range(1, 8):
                    _in_fn = in_fn[:in_fn.rfind('.')] +  '_' + str(i) + in_fn[in_fn.rfind('.'):]
                    _llpm_fn = in_fn[:in_fn.rfind('.')] + '_llpm_' + str(i) + in_fn[in_fn.rfind('.'):]
                    if (not os.path.isfile(_llpm_fn) or overwrite):
                        _in = np.load(_in_fn).astype(np.float32)
                        _in = np.where(np.isfinite(_in), _in, 1.0e+38)
                        _in = np.where(_in < 1.0e+38, _in, 1.0e+38)

                        llpm_buffer = self._preprocess_llpm(_in)
                        np.save(_llpm_fn, llpm_buffer)
            
            if self.mode != 'test' :
                sbmc_s_buffer, sbmc_p_buffer = self._preprocess_sbmc(sample)
            
            if (sbmc):    
                if (not os.path.isfile(sbmc_s_fn) or overwrite):
                    np.save(sbmc_s_fn, sbmc_s_buffer)
                
                if (not os.path.isfile(sbmc_p_fn) or overwrite):
                    np.save(sbmc_p_fn, sbmc_p_buffer)
            
            if (kpcn):
                if self.mode == 'test':
                    for _s in [2, 4, 8, 16, 32, 64]: # support upto 64 spp due to the lack of gpu memory
                        kpcn_fn = in_fn[:in_fn.rfind('.')] +  '_kpcn_' + str(_s) + in_fn[in_fn.rfind('.'):]
                        
                        if (not os.path.isfile(kpcn_fn) or overwrite):
                            _in = sample
                            s = _in.shape[2]

                            i = 0
                            while s < _s:
                                i += 1
                                _in_fn = in_fn[:in_fn.rfind('.')] +  '_' + str(i) + in_fn[in_fn.rfind('.'):]
                                assert os.path.isfile(_in_fn), 'Too many number of samples. %d spp is not supported by %s.'%(_s, _in_fn)
                                _in2 = np.load(_in_fn, mmap_mode='r').astype(np.float32)
                                _in2 = np.where(np.isfinite(_in2), _in2, 1.0e+38)
                                _in2 = np.where(_in2 < 1.0e+38, _in2, 1.0e+38)
                                s2 = _in2.shape[2]
                                s += s2
                                _in = np.concatenate((_in, _in2), axis=2)
                            
                            #print(_in[:,:,:_s,:].shape)
                            kpcn_buffer = self._preprocess_kpcn(_in[:,:,:_s,:])
                            np.save(kpcn_fn, kpcn_buffer)

                else:
                    for _s in range(2, self.spp + 1):
                        assert self.mode != 'test'
                        kpcn_fn = in_fn[:in_fn.rfind('.')] +  '_kpcn_' + str(_s) + in_fn[in_fn.rfind('.'):]

                        if (not os.path.isfile(kpcn_fn) or overwrite):
                            kpcn_buffer = self._preprocess_kpcn(sample[:,:,:_s,:])
                            np.save(kpcn_fn, kpcn_buffer)

            if (remove_original):
                os.remove(in_fn)
            
            # Target preprocess
            _gt = np.load(gt_fn).astype(np.float32)
            _gt = np.where(np.isfinite(_gt), _gt, 1.0e+38)
            _gt = np.where(_gt < 1.0e+38, _gt, 1.0e+38)

            np.save(gt_fn, _gt)

            if self.mode != 'test':
                # Patch sampling map
                if (not os.path.isfile(prob_fn) or overwrite):
                    gt = LinearToSrgb(ToneMap(_gt[...,:3], 1.5)) # intended mistaken
                    diffuse = sbmc_p_buffer[...,75-27].mean(2) # 75-27 72-27
                    glossy = sbmc_p_buffer[...,81-27].mean(2) # 81-27 78-27
                    specular = sbmc_p_buffer[...,87-27].mean(2) # 87-27 84-27
                    normal = sbmc_s_buffer[...,20:23].mean(2) * 0.5 + 0.5 # 20:23 20:23
                    
                    lum = 0.2126 * gt[:,:,0] + 0.7152 * gt[:,:,1] + 0.0722 * gt[:,:,2]
                    d_lum = gradient_importance_map(lum)
                    d_norm = gradient_importance_map(normal)
                    mat = (diffuse + glossy * 4 + specular * 2) / 7

                    prob = 0.3 * d_lum + 0.2 * d_norm + 0.5 * mat
                    h, w = prob.shape
                    prob = prob[self.PATCH_SIZE//2:-self.PATCH_SIZE//2,self.PATCH_SIZE//2:-self.PATCH_SIZE//2]
                    prob /= (np.sum(prob) + 1e-5)
                    np.save(prob_fn, prob)

# Sampling
    def _random_rot(self, sample):
        """
        Args:
            sample(dict) with the following key candidates:
                "paths"
                "features"
                "radiance"
                "target_image"
                // No global features
        """
        k = random.randrange(0, 4)
        sample["paths"] = np.rot90(sample["paths"], k).copy()
        sample["features"] = np.rot90(sample["features"], k).copy()
        sample["radiance"] = np.rot90(sample["radiance"], k).copy()
        sample["target_image"] = np.rot90(sample["target_image"], k).copy()
        
        return sample

    def _random_flip(self, sample):
        """
        Args:
            sample(dict) with the following key candidates:
                "paths"
                "features"
                "radiance"
                "target_image"
                // No global features
        """
        if random.randrange(0, 2):
            sample["paths"] = np.flipud(sample["paths"]).copy()
            sample["features"] = np.flipud(sample["features"]).copy()
            sample["radiance"] = np.flipud(sample["radiance"]).copy()
            sample["target_image"] = np.flipud(sample["target_image"]).copy()
        
        if random.randrange(0, 2):
            sample["paths"] = np.fliplr(sample["paths"]).copy()
            sample["features"] = np.fliplr(sample["features"]).copy()
            sample["radiance"] = np.fliplr(sample["radiance"]).copy()
            sample["target_image"] = np.fliplr(sample["target_image"]).copy()
        
        return sample

    def _transpose(self, sample):     
        if type(sample) == tuple:
            assert len(sample) == 2, 'behavior undefined.: %f'%(len(sample))

            sample_1, sample_2 = sample

            for k in sample_1:
                if type(sample_1[k]) == np.ndarray or type(sample_1[k]) == np.memmap:
                    if (len(sample_1[k].shape) == 3):
                        sample_1[k] = np.transpose(sample_1[k], (2, 0, 1))
                    elif (len(sample_1[k].shape) == 4):
                        sample_1[k] = np.transpose(sample_1[k], (2, 3, 0, 1))
                    else:
                        assert False, 'behavior undefined.'
            
            for k in sample_2:
                if type(sample_2[k]) == np.ndarray or type(sample_2[k]) == np.memmap:
                    if (len(sample_2[k].shape) == 3):
                        sample_2[k] = np.transpose(sample_2[k], (2, 0, 1))
                    elif (len(sample_2[k].shape) == 4):
                        sample_2[k] = np.transpose(sample_2[k], (2, 3, 0, 1))
                    else:
                        assert False, 'behavior undefined.'
        else:
            for k in sample:
                if type(sample[k]) == np.ndarray or type(sample[k]) == np.memmap:
                    if (len(sample[k].shape) == 3):
                        sample[k] = np.transpose(sample[k], (2, 0, 1))
                    elif (len(sample[k].shape) == 4):
                        sample[k] = np.transpose(sample[k], (2, 3, 0, 1))
                    else:
                        assert False, 'behavior undefined.'

        return sample

    def _sample_patches(self, sample, prob):
        """Sample patches according to their importance
        """
        self.samples = []
        h, w = prob.shape
        prob = prob.reshape(h*w)

        # Sample the regions of interest
        try:
            roi = np.random.choice(h*w, size=self.patches_per_image, p=prob)
        except ValueError:
            roi = np.random.choice(h*w, size=self.patches_per_image)
        
        for idx in roi:
            x = idx // w
            y = idx % w

            if type(sample) == tuple:
                assert len(sample) == 2, 'behavior undefined.: %f'%(len(sample))

                patch_1 = {}
                patch_2 = {}
                sample_1, sample_2 = sample

                for k in sample_1:
                    if type(sample_1[k]) == np.ndarray or type(sample_1[k]) == np.memmap:
                        patch_1[k] = sample_1[k][x:x+self.PATCH_SIZE,y:y+self.PATCH_SIZE,...]
                    else:
                        patch_1[k] = sample_1[k]
                
                for k in sample_2:
                    if type(sample_2[k]) == np.ndarray or type(sample_2[k]) == np.memmap:
                        patch_2[k] = sample_2[k][x:x+self.PATCH_SIZE,y:y+self.PATCH_SIZE,...]
                    else:
                        patch_2[k] = sample_2[k]
                
                self.samples.append((patch_1, patch_2))
            else:
                patch = {}
                for k in sample:
                    if type(sample[k]) == np.ndarray or type(sample[k]) == np.memmap:
                        patch[k] = sample[k][x:x+self.PATCH_SIZE,y:y+self.PATCH_SIZE,...]
                    else:
                        patch[k] = sample[k]
                
                self.samples.append(patch)
    
    def _full_patches(self, sample):
        """Return all grid patches
        """
        self.samples = []
        if self.base_model == 'sbmc':
            h, w, _, = sample['target_image'].shape
        elif self.base_model == 'kpcn':
            h, w, _, = sample['target_diffuse'].shape
        elif self.base_model == 'amcd':
            h, w, _, = sample[0]['ref'].shape

        for x in range(0, h, self.PATCH_SIZE):
            for y in range(0, w, self.PATCH_SIZE):
                if type(sample) == tuple:
                    assert len(sample) == 2, 'behavior undefined.: %f'%(len(sample))

                    patch_1 = {}
                    patch_2 = {}
                    sample_1, sample_2 = sample

                    for k in sample_1:
                        if type(sample_1[k]) == np.ndarray or type(sample_1[k]) == np.memmap:
                            patch_1[k] = sample_1[k][x:x+self.PATCH_SIZE,y:y+self.PATCH_SIZE,...]
                        else:
                            patch_1[k] = sample_1[k]
                    
                    for k in sample_2:
                        if type(sample_2[k]) == np.ndarray or type(sample_2[k]) == np.memmap:
                            patch_2[k] = sample_2[k][x:x+self.PATCH_SIZE,y:y+self.PATCH_SIZE,...]
                        else:
                            patch_2[k] = sample_2[k]

                    self.samples.append((patch_1, patch_2))
                else:
                    patch = {}
                    for k in sample:
                        if type(sample[k]) == np.ndarray or type(sample[k]) == np.memmap:
                            patch[k] = sample[k][x:x+self.PATCH_SIZE,y:y+self.PATCH_SIZE,...]
                        else:
                            patch[k] = sample[k]
                    
                    self.samples.append(patch)
        # print('PATCH NUM FOR VAL SCENE : {}'.format(len(self.samples)))

    def _recon_patches(self, sample):
        """Return all grid patches
        """
        self.samples = []
        if self.base_model == 'sbmc':
            h, w, _, = sample['target_image'].shape
        elif self.base_model == 'kpcn':
            h, w, _, = sample['target_diffuse'].shape
            # print(h, w)
        elif self.base_model == 'amcd':
            h, w, _, = sample[0]['ref'].shape

        # Suppose that self.PATCH_SIZE is 128
        sample_num = len(self.samples)
        for x in range(0, h-127, self.PATCH_SIZE // 2):
            for y in range(0, w-127, self.PATCH_SIZE // 2):
                if type(sample) == tuple:
                    assert len(sample) == 2, 'behavior undefined.: %f'%(len(sample))

                    patch_1 = {}
                    patch_2 = {}
                    sample_1, sample_2 = sample

                    for k in sample_1:
                        if type(sample_1[k]) == np.ndarray or type(sample_1[k]) == np.memmap:
                            patch_1[k] = sample_1[k][x:x+self.PATCH_SIZE,y:y+self.PATCH_SIZE,...]
                        else:
                            patch_1[k] = sample_1[k]
                    
                    for k in sample_2:
                        if type(sample_2[k]) == np.ndarray or type(sample_2[k]) == np.memmap:
                            patch_2[k] = sample_2[k][x:x+self.PATCH_SIZE,y:y+self.PATCH_SIZE,...]
                        else:
                            patch_2[k] = sample_2[k]

                    self.samples.append((patch_1, patch_2))
                else:
                    patch = {}
                    for k in sample:
                        if type(sample[k]) == np.ndarray or type(sample[k]) == np.memmap:
                            patch[k] = sample[k][x:x+self.PATCH_SIZE,y:y+self.PATCH_SIZE,...]
                        else:
                            patch[k] = sample[k]
                    
                    self.samples.append(patch)
        # print('MADE {} PATCHES'.format(len(self.samples)-sample_num))

# Statistics
    def _load_raw_data(self, img_idx):
        """
        Returns:
            sample(dict) with the following keys:
                "input_samples"
                "target_images"
        """
        sample = {}
        in_fn = self.gt_files[img_idx].replace('gt', 'input')
        gt_fn = self.gt_files[img_idx]

        # Load input sample radiances
        _in = np.load(in_fn)[:,:,:self.spp,:].astype(np.float32)

        # Load target images
        _gt = np.load(gt_fn).astype(np.float32)

        # NaN handling
        _in = np.where(np.isfinite(_in), _in, 1.0e+38)
        _in = np.where(_in < 1.0e+38, _in, 1.0e+38)
        _gt = np.where(np.isfinite(_gt), _gt, 1.0e+38)
        _gt = np.where(_gt < 1.0e+38, _gt, 1.0e+38)

        sample['input_samples'] = _in
        sample['target_images'] = _gt

        return sample

    def get_stats(self):
        """Return the sample mean and sample standard deviation from the dataset. Use those values to data standardization. Try this function before training using your custom dataset.
        Note: Data standardization for tr/val/test datasets must be done by 
              using training mean and std., not validation nor testing samples.
              Also, "radiance", "kpcn_diffuse_buffer", "kpcn_specular_buffer", 
              "target_image" features should not be standardized.
        
        Returns:
            a tuple of the sample mean and sample standard deviation.
        """
        if len(self.gt_files) == 0:
            raise RuntimeError("No data assigned for the `DenoiseDataset` object")

        def size_by_axes(shape, axis):
            if len(axis) == 0:
                raise RuntimeError("Invalid axes configuration")
            if len(shape) == 0:
                raise RuntimeError("Invalid shape configuration")

            s = 1
            for a in axis:
                s *= shape[a]
            return s
        
        sz_fet = 0
        sz_pth = 0

        for img_idx in tqdm(range(len(self.gt_files)), leave=False, ncols=100):
            sample = self._load_raw_data(img_idx)

            if self.base_model == self.SBMC:
                sample = self._preprocess_sbmc(sample)
            elif self.base_model == self.KPCN:
                sample = self._preprocess_kpcn(sample)
            else:
                raise RuntimeError("Unknown baseline model %s" % self.base_model)
            
            # assume the size of every data points is the same
            # batch mean
            bm_fet = sample["features"].mean((0,1,2))
            bm_pth = sample["paths"].mean((0,1,2))
            
            # batch unbiased variance
            bv_fet = sample["features"].var((0,1,2), ddof=1)
            bv_pth = sample["paths"].var((0,1,2), ddof=1)

            # batch sample size
            bsz_fet = size_by_axes(sample["features"].shape, (0,1,2))
            bsz_pth = size_by_axes(sample["paths"].shape, (0,1,2))

            bv_fet *= (bsz_fet - 1)
            bv_pth *= (bsz_pth - 1)

            sz_fet += bsz_fet
            sz_pth += bsz_pth

            if img_idx == 0:
                # for the sample mean
                m_fet = bm_fet
                m_pth = bm_pth
                
                # for the unbiased sample variance
                # Note: the sqrt of the unbiased sample variance is yet biased.
                # See https://en.wikipedia.org/wiki/Standard_deviation#Sample_standard_deviation
                v_fet = bv_fet
                v_pth = bv_pth

                max_fet = sample["features"].max(axis=(0,1,2))
                max_pth = sample["paths"].max(axis=(0,1,2))

                min_fet = sample["features"].min(axis=(0,1,2))
                min_pth = sample["paths"].min(axis=(0,1,2))
            else:
                m_fet += (bm_fet - m_fet) / (img_idx + 1)
                m_pth += (bm_pth - m_pth) / (img_idx + 1)

                v_fet += (bv_fet + (img_idx + 1) / img_idx * bsz_fet * (m_fet - bm_fet) ** 2)
                v_pth += (bv_pth + (img_idx + 1) / img_idx * bsz_pth * (m_pth - bm_pth) ** 2)

                max_fet = np.maximum(sample["features"].max(axis=(0,1,2)), max_fet)
                max_pth = np.maximum(sample["paths"].max(axis=(0,1,2)), max_pth)

                min_fet = np.minimum(sample["features"].min(axis=(0,1,2)), min_fet)
                min_pth = np.minimum(sample["paths"].min(axis=(0,1,2)), min_pth)

        mean = {
            'features': m_fet,
            'path': m_pth
        }

        std = {
            'features': np.sqrt(v_fet / (sz_fet - 1)),
            'path': np.sqrt(v_pth / (sz_pth - 1))
        }

        M = {
            'features': max_fet,
            'path': max_pth
        }

        m = {
            'features': min_fet,
            'path': min_pth
        }

        print(mean)
        print(std)
        print(M)
        print(m)

        return mean, std, M, m

# Get item
    def __getitem__(self, idx):
        """
        Returns:
            sample(dict) with the following key candidates:
                "paths"
                "features"
                "radiance"
                "target_image"
                // No global features
        """
        
        img_idx = idx // self.patches_per_image + 0
        pat_idx = idx % self.patches_per_image
        # print('get item : {}, {}'.format(idx, pat_idx))
        if (pat_idx == 0):
            sample = {}
            
            # Input processing
            in_fn = self.gt_files[img_idx].replace(os.sep + 'gt' + os.sep, os.sep + 'input' + os.sep)
            # print(in_fn.split('/')[-1])
            
            if (self.base_model == self.SBMC):
                sbmc_s_fn = in_fn[:in_fn.rfind('.')] + '_sbmc_s' + in_fn[in_fn.rfind('.'):]
                sbmc_p_fn = in_fn[:in_fn.rfind('.')] + '_sbmc_p' + in_fn[in_fn.rfind('.'):]
                sbmc_s_fn = sbmc_s_fn.replace(os.sep + 'KPCN' + os.sep, os.sep + 'SBMC' + os.sep)
                sbmc_p_fn = sbmc_p_fn.replace(os.sep + 'KPCN' + os.sep, os.sep + 'SBMC' + os.sep)
                sbmc_s_fn = get_valid_path(sbmc_s_fn)
                sbmc_p_fn = get_valid_path(sbmc_p_fn)

                _in_s = np.load(sbmc_s_fn, mmap_mode='r')[...,:self.spp,:]
                p_buf = np.load(sbmc_p_fn, mmap_mode='r')[...,:self.spp,:]

                total_rad = np.array(_in_s[...,:3])

                if (self.use_g_buf and self.use_sbmc_buf):
                    s_buf = _in_s[...,3:3+24]
                    sample['radiance'] = total_rad
                    sample['features'] = np.concatenate([s_buf, p_buf], axis=3)
                elif (self.use_g_buf):
                    s_buf = _in_s[...,3:3+24]
                    sample['radiance'] = total_rad
                    sample['features'] = np.array(s_buf)
                elif (self.use_sbmc_buf):
                    total = _in_s[...,3:3+3]
                    sample['radiance'] = total_rad
                    sample['features'] = np.concatenate([total, p_buf], axis=3)
                else:
                    total = _in_s[...,3:3+3]
                    sample['radiance'] = total_rad
                    sample['features'] = np.array(total)
            elif (self.base_model == self.KPCN):
                kpcn_fn = in_fn[:in_fn.rfind('.')] + '_kpcn_' + str(self.spp) + in_fn[in_fn.rfind('.'):]
                kpcn_fn = get_valid_path(kpcn_fn)

                _in = np.load(kpcn_fn)
                # print(np.shape(_in))
                
                sample['kpcn_diffuse_in'] = np.concatenate([_in[...,:10], _in[...,20:]], axis=2)
                sample['kpcn_specular_in'] = _in[...,10:]
                sample['kpcn_diffuse_buffer'] = _in[...,:3]
                sample['kpcn_specular_buffer'] = _in[...,10:13]
                sample['kpcn_albedo'] = _in[...,34:37] + 0.00316 # Note
            elif (self.base_model == self.AMCD):
                kpcn_fn = in_fn[:in_fn.rfind('.')] + '_kpcn_' + str(self.spp) + in_fn[in_fn.rfind('.'):]

                _in = np.load(kpcn_fn)

                diffuse_sample = {
                    'in': _in[...,:3],
                    'seg': np.concatenate([_in[...,20:23], _in[...,30:31], _in[...,34:37]], axis=2),
                    'x_offset': self.PATCH_SIZE,
                    'y_offset': self.PATCH_SIZE
                }

                specular_sample = {
                    'in': _in[...,10:13],
                    'seg': np.concatenate([_in[...,20:23], _in[...,30:31], _in[...,34:37]], axis=2),
                    'x_offset': self.PATCH_SIZE,
                    'y_offset': self.PATCH_SIZE
                }

            if (self.use_llpm_buf):
                llpm_fn = in_fn[:in_fn.rfind('.')] + '_llpm' + in_fn[in_fn.rfind('.'):]
                llpm_fn = llpm_fn.replace(os.sep + 'KPCN' + os.sep, os.sep + 'LLPM' + os.sep)
                llpm_fn = get_valid_path(llpm_fn)

                _in = np.load(llpm_fn, mmap_mode='r')[...,:self.spp,:]

                # Path sampling weight
                if (self.base_model == self.SBMC):
                    sample['features'] = np.concatenate((
                        sample['features'], 
                        _in[...,:1]
                        ), axis=3)
                elif (self.base_model == self.KPCN):
                    sample['kpcn_diffuse_in'] = np.concatenate((
                        sample['kpcn_diffuse_in'], 
                        _in[...,:1].mean(2)
                        ), axis=2)
                    sample['kpcn_specular_in'] = np.concatenate((
                        sample['kpcn_specular_in'], 
                        _in[...,:1].mean(2)
                        ), axis=2)
                elif (self.base_model == self.AMCD):
                    diffuse_sample['seg'] = np.concatenate((
                        diffuse_sample['seg'], 
                        _in[...,:1].mean(2)
                        ), axis=2)
                    specular_sample['seg'] = np.concatenate((
                        specular_sample['seg'], 
                        _in[...,:1].mean(2)
                        ), axis=2)

                # Path descriptor
                if (self.base_model == self.AMCD):
                    diffuse_sample['paths'] = _in[...,1:] # prevent redundant GRAM use
                else:
                    sample['paths'] = np.array(_in[...,1:])

            # print('Push Data')
            # Target processing
            gt_fn = get_valid_path(self.gt_files[img_idx])

            _gt = np.load(gt_fn)

            if (self.base_model == self.SBMC):
                sample['target_image'] = _gt[:,:,0:3]
            elif (self.base_model == self.KPCN):
                total = _gt[:,:,0:3]
                diffuse = _gt[:,:,3:6]
                albedo = _gt[:,:,6:]

                sample['target_diffuse'] = diffuse / (albedo + 0.00316)
                sample['target_specular'] = np.log(1 + total - diffuse)
                sample['target_total'] = total
            elif (self.base_model == self.AMCD):
                total = _gt[:,:,0:3]
                diffuse = _gt[:,:,3:6]
                albedo = _gt[:,:,6:]

                diffuse_sample['ref'] = diffuse / (albedo + 0.00316)
                specular_sample['ref'] = np.log(1 + total - diffuse)

            # # Patch sampling probability map
            # prob_fn = in_fn[:in_fn.rfind('.')] + '_prob_imp' + in_fn[in_fn.rfind('.'):]
            # prob_fn = get_valid_path(prob_fn)

            # prob_map = np.load(prob_fn)

          
            if self.sampling == 'random':
                # Patch sampling probability map
                prob_fn = in_fn[:in_fn.rfind('.')] + '_prob_imp' + in_fn[in_fn.rfind('.'):]
                prob_fn = get_valid_path(prob_fn)

                prob_map = np.load(prob_fn)
                if (self.base_model == self.AMCD):
                    self._sample_patches((diffuse_sample, specular_sample), prob_map)
                else:
                    
                    self._sample_patches(sample, prob_map)
            elif self.sampling == 'grid':
                if (self.base_model == self.AMCD):
                    self._full_patches((diffuse_sample, specular_sample))
                else:
                    self._full_patches(sample)
            elif self.sampling == 'recon':
                if (self.base_model == self.AMCD):
                    self._recon_patches((diffuse_sample, specular_sample))
                else:
                    self._recon_patches(sample)
            else:
                raise RuntimeError("Unknown training mode %s" % self.mode)
        if self.samples != None: len(self.samples)
        out = self.samples[pat_idx]
        # if self.sampling == 'random':
        # out = self._random_rot(self._random_flip(out))
        out = self._transpose(out)

        return out


class MSDenoiseDataset(ConcatDataset):
    
    """
    Multi sample count version of the `DenoiseDataset` class.
    """

    def __init__(self, dir, spp, base_model='sbmc', mode='train', batch_size=8,
                 sampling='random', use_g_buf=True, use_sbmc_buf=True, 
                 use_llpm_buf=False, pnet_out_size=3):
        if spp < 2:
            raise RuntimeError("spp too low to randomize sample count, should"
                               "be at least 2.")
        datasets = []
        for _s in range(2, spp + 1):
            datasets.append(
                DenoiseDataset(dir, _s, base_model, mode, batch_size,
                               sampling, use_g_buf, use_sbmc_buf, use_llpm_buf, pnet_out_size)
            )
            # print(datasets[-1].samples)
        super(MSDenoiseDataset, self).__init__(datasets)

        self.dncnn_in_size = datasets[0].dncnn_in_size
        self.pnet_out_size = datasets[0].pnet_out_size
        self.pnet_in_size = datasets[0].pnet_in_size


class FullImageDataset(Dataset):

    """Mode to preprocess features to the format expected by [Gharbi2019]"""
    SBMC = "sbmc"
    """Mode to preprocess features to the format expected by [Munkberg2020]"""
    LBMC = "lbmc"
    """Mode to preprocess features to the format expected by [Bako2017]"""
    KPCN = "kpcn"
    """Mode to preprocess features to the format expected by [Xu2019]"""
    AMCD = "amcd"

    MAX_DEPTH = 5

    PATCH_SIZE = 128

    def __init__(self, in_fn, spp, base_model='sbmc', use_g_buf=True, 
                 use_sbmc_buf=True, use_llpm_buf=False, pnet_out_size=3, visualize=False, feat_imp=False):
        if base_model not in [self.KPCN, self.SBMC, self.AMCD, self.LBMC]:
            raise RuntimeError("Unknown baseline model %s" % base_model)
        assert os.sep + 'input' + os.sep in in_fn, in_fn

        if base_model == self.LBMC:
            base_model = self.SBMC
            use_sbmc_buf = False
            use_g_buf = True
        
        self.in_fn = in_fn
        self.gt_fn = in_fn.replace(os.sep + 'input' + os.sep, os.sep + 'gt' + os.sep)
        self.spp = spp
        self.base_model = base_model
        self.use_g_buf = use_g_buf
        self.use_sbmc_buf = use_sbmc_buf
        self.use_llpm_buf = use_llpm_buf
        self.samples = []
        self.coords = []
        patch_size = self.PATCH_SIZE
        pad_size = 32

        # Model input channel size
        self.pnet_in_size = 0
        if use_llpm_buf:
            self.pnet_in_size += 36
        
        self.pnet_out_size = pnet_out_size
        
        self.dncnn_in_size = 0
        if base_model == self.SBMC:
            self.dncnn_in_size = 3
            if use_g_buf:
                self.dncnn_in_size += 21
            if use_sbmc_buf:
                self.dncnn_in_size += 66
        elif base_model == self.KPCN:
            self.dncnn_in_size = 34
        elif base_model == self.AMCD:
            self.dncnn_in_size = 10

        if use_llpm_buf:
            self.dncnn_in_size += pnet_out_size + 2 # path weight, p-buffer, variance
        
        sample = self._load_full_buffer()
        
        if base_model == self.KPCN:
            h, w, _ = sample['target_total'].shape
            self.h, self.w = h, w
            self.has_hit = np.concatenate((self.has_hit,)*3, axis=2)
            self.full_ipt = sample['kpcn_diffuse_buffer'] * sample['kpcn_albedo'] + np.exp(sample['kpcn_specular_buffer']) - 1
            self.full_tgt = sample['target_total']
            assert self.has_hit.shape[-1] == 3, self.has_hit.shape

            if visualize:
                self.normal = 0.5 * sample['kpcn_diffuse_in'][...,10:13] + 0.5
                self.depth = sample['kpcn_diffuse_in'][...,20]
                self.albedo = sample['kpcn_diffuse_in'][...,24:27]
        elif base_model == self.SBMC:
            h, w, _, _ = sample['radiance'].shape
            self.h, self.w = h, w
            self.has_hit = np.concatenate((self.has_hit,)*3, axis=2)
            self.full_ipt = np.mean(sample['radiance'], 2)
            self.full_tgt = sample['target_image']

        if visualize and use_llpm_buf:
            def mapping(img):
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                img = LinearToSrgb(ToneMap(img))
                return img

            self.radiance_wo_weight = mapping(np.mean(sample['paths'], 2)[...,:3])
            self.light_intensity = mapping(np.mean(sample['paths'], 2)[...,3:6])
            self.throughput = mapping(np.mean(sample['paths'], 2)[...,6:9])
            self.bounce_type = np.mean(sample['paths'], 2)[...,24]
            self.roughness = np.mean(sample['paths'], 2)[...,30]
        
        if use_llpm_buf and feat_imp:
            h, w, s, c = sample['paths'].shape
            idx = torch.randperm(h * w * s)

            _p = sample['paths'].reshape(h * w * s, c)
            _p = _p[idx, :]
            sample['paths'] = _p.reshape(h, w, s, c)

        for k in sample:
            if len(sample[k].shape) == 3:
                sample[k] = sample[k].transpose([2, 0, 1])
            elif len(sample[k].shape) == 4:
                sample[k] = sample[k].transpose([2, 3, 0, 1])

        stride = patch_size - 2 * pad_size
        assert (h - 2 * pad_size) % stride == 0 and (w - 2 * pad_size) % stride == 0
        
        for i in range(0, h - 2 * pad_size, stride):
            for j in range(0, w - 2 * pad_size, stride):
                i_start = i + pad_size
                j_start = j + pad_size
                i_end = i + patch_size - pad_size
                j_end = j + patch_size - pad_size

                if i == 0:
                    i_start = 0
                if j == 0:
                    j_start = 0
                if i == h - patch_size:
                    i_end = i + patch_size
                if j == w - patch_size:
                    j_end = j + patch_size
                self.coords.append((i_start, j_start, i_end, j_end, i, j))

                patch = {}
                for k in sample:
                    patch[k] = sample[k][...,i:i+patch_size,j:j+patch_size]
                self.samples.append(patch)
    
    def _load_all_spp_buffer(self, base_fn):
        assert base_fn.endswith('.npy'), base_fn
        _in = np.load(base_fn, mmap_mode='r')
        _, _, s, _ = _in.shape
        i = 0
        
        while s < self.spp:
            i += 1
            _base_fn = base_fn[:-4] + '_' + str(i) + '.npy'
            _in2 = np.load(_base_fn, mmap_mode='r')
            _, _, s2, _ = _in2.shape
            s += s2
            _in =np.concatenate((_in, _in2), axis=2)
        _in = _in[...,:self.spp,:]
        
        return _in

    def _load_full_buffer(self):
        in_fn = self.in_fn
        sample = {}

        if (self.base_model == self.SBMC):
            sbmc_s_fn = in_fn[:in_fn.rfind('.')] + '_sbmc_s' + in_fn[in_fn.rfind('.'):]
            sbmc_p_fn = in_fn[:in_fn.rfind('.')] + '_sbmc_p' + in_fn[in_fn.rfind('.'):]
            sbmc_s_fn = sbmc_s_fn.replace(os.sep + 'KPCN' + os.sep, os.sep + 'SBMC' + os.sep)
            sbmc_p_fn = sbmc_p_fn.replace(os.sep + 'KPCN' + os.sep, os.sep + 'SBMC' + os.sep)
            sbmc_s_fn = get_valid_path(sbmc_s_fn)
            sbmc_p_fn = get_valid_path(sbmc_p_fn)

            _in_s = self._load_all_spp_buffer(sbmc_s_fn)
            p_buf = self._load_all_spp_buffer(sbmc_p_fn)

            total_rad = np.array(_in_s[...,:3])

            if (self.use_g_buf and self.use_sbmc_buf):
                s_buf = _in_s[...,3:3+24]
                sample['radiance'] = total_rad
                sample['features'] = np.concatenate([s_buf, p_buf], axis=3)
            elif (self.use_g_buf):
                s_buf = _in_s[...,3:3+24]
                sample['radiance'] = total_rad
                sample['features'] = np.array(s_buf)
            elif (self.use_sbmc_buf):
                total = _in_s[...,3:3+3]
                sample['radiance'] = total_rad
                sample['features'] = np.concatenate([total, p_buf], axis=3)
            else:
                total = _in_s[...,3:3+3]
                sample['radiance'] = total_rad
                sample['features'] = np.array(total)
        elif (self.base_model == self.KPCN):
            kpcn_fn = in_fn[:in_fn.rfind('.')] + '_kpcn_' + str(self.spp) + in_fn[in_fn.rfind('.'):]
            kpcn_fn = get_valid_path(kpcn_fn)
            
            _in = np.load(kpcn_fn)
            
            sample['kpcn_diffuse_in'] = np.concatenate([_in[...,:10], _in[...,20:]], axis=2)
            sample['kpcn_specular_in'] = _in[...,10:]
            sample['kpcn_diffuse_buffer'] = _in[...,:3]
            sample['kpcn_specular_buffer'] = _in[...,10:13]
            sample['kpcn_albedo'] = _in[...,34:37] + 0.00316

        if (self.use_llpm_buf):
            llpm_fn = in_fn[:in_fn.rfind('.')] + '_llpm' + in_fn[in_fn.rfind('.'):]
            llpm_fn = llpm_fn.replace(os.sep + 'KPCN' + os.sep, os.sep + 'LLPM' + os.sep)
            llpm_fn = get_valid_path(llpm_fn)
            
            _in = self._load_all_spp_buffer(llpm_fn)

            # Path sampling weight
            if (self.base_model == self.SBMC):
                sample['features'] = np.concatenate((
                    sample['features'], 
                    _in[...,:1]
                    ), axis=3)
            elif (self.base_model == self.KPCN):
                sample['kpcn_diffuse_in'] = np.concatenate((
                    sample['kpcn_diffuse_in'], 
                    _in[...,:1].mean(2)
                    ), axis=2)
                sample['kpcn_specular_in'] = np.concatenate((
                    sample['kpcn_specular_in'], 
                    _in[...,:1].mean(2)
                    ), axis=2)
            
            # Path descriptor
            sample['paths'] = np.array(_in[...,1:])
        
        # Target processing
        gt_fn = get_valid_path(self.gt_fn)

        _gt = np.load(gt_fn)

        if (self.base_model == self.SBMC):
            sample['target_image'] = _gt[:,:,0:3]
        elif (self.base_model == self.KPCN):
            total = _gt[:,:,0:3]
            diffuse = _gt[:,:,3:6]
            albedo = _gt[:,:,6:]

            sample['target_diffuse'] = diffuse / (albedo + 0.00316)
            sample['target_specular'] = np.log(1 + total - diffuse)
            sample['target_total'] = total

        # Post-processing feature (NOTE: dirty code)
        llpm_fn = in_fn[:in_fn.rfind('.')] + '_llpm' + in_fn[in_fn.rfind('.'):]
        llpm_fn = llpm_fn.replace(os.sep + 'KPCN' + os.sep, os.sep + 'LLPM' + os.sep)
        llpm_fn = get_valid_path(llpm_fn)
        
        _in = np.load(llpm_fn, mmap_mode='r')
        # the first bounce type = 0 means that an eye ray hits the background or an emitter
        # this can be easily replaced by first bounce normal vector or hitBuffer or else.
        self.has_hit = (np.mean(np.array(_in[...,1:]), 2)[...,24:25] != 0.0).astype(np.float32) 

        return sample

    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        i_start, j_start, i_end, j_end, i, j = self.coords[idx]
        out = self.samples[idx]

        return out, i_start, j_start, i_end, j_end, i, j


def init_data(args):
    # Initialize datasets
    print('init_data')
    datasets = {}
    # datasets['train'] = MSDenoiseDataset(args.data_dir, 8, 'kpcn', 'train', args.batch_size, 'random',
    #     use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, pnet_out_size=3)
    datasets['train'] = DenoiseDataset(args.data_dir, 8, 'kpcn', 'train', args.batch_size, 'random',
      use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, pnet_out_size=3)
    # print('DATASET sample length ', len(datasets['train'].samples))
    # datasets['val'] = MSDenoiseDataset(args.data_dir, 8, 'kpcn', 'val', 4, 'grid',
    #     use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, pnet_out_size=3)
    datasets['val'] = DenoiseDataset(args.data_dir, 8, 'kpcn', 'val', 4, 'grid',
        use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, pnet_out_size=3)
    # datasets['val'] = DenoiseDataset(args.data_dir, 8, 'kpcn', 'train', 4, 'grid',
    #     use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, pnet_out_size=3)
    
    # Initialize dataloaders
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        datasets['train'], 
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=False,
    )
    dataloaders['val'] = DataLoader(
        datasets['val'],
        batch_size=4,
        num_workers=1,
        pin_memory=False
    )
    return datasets, dataloaders
    


if __name__ == '__main__':
    """
    Test codes
    Note: comment out the `out = self._transpose(out)`
    """
    def test1():
        dataset = DenoiseDataset('D:\\LLPM', 8, mode='val', base_model='kpcn', use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, sampling='grid')
        start = time.time()
        dataset._offline_preprocess(sbmc=False, llpm=True, kpcn=True, overwrite=False)
        print(time.time() - start)
    
    def test2():
        dataset = DenoiseDataset('/mnt/ssd3/iycho/KPCN', 8, mode='val', base_model='kpcn', use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, sampling='random')
        start = time.time()
        dataset.__getitem__(0)
        print(time.time() - start)
    
    def test3():
        dataset = FullImageDataset('D:\\LLPM_data\\val\\input\\material-testball_28.npy', 8, 'kpcn', True, False, False, 3)
        plt.title('has hit')
        plt.imshow(dataset.has_hit)
        plt.show()

    def save_all_buffers():
        for s in ['bathroom', 'bathroom-3', 'car', 'car2', 'chair', 'chair-room', 'gharbi', 'hookah', 'kitchen-2', 'library-office', 'path-manifold', 'sitting-room-2', 'tableware', 'teapot']:
            dataset = FullImageDataset('D:\\LLPM\\test\\input\\%s.npy'%(s), 8, 'kpcn', True, False, True, 3, visualize=True)
            plt.imsave('C:\\Users\\dorian\\Desktop\\%s-normal.png'%(s), dataset.normal)
            plt.imsave('C:\\Users\\dorian\\Desktop\\%s-depth.png'%(s), dataset.depth)
            plt.imsave('C:\\Users\\dorian\\Desktop\\%s-albedo.png'%(s), dataset.albedo)

            plt.imsave('C:\\Users\\dorian\\Desktop\\%s-radiance_wo_weight.png'%(s), dataset.radiance_wo_weight, vmin=0.0, vmax=1.0)
            plt.imsave('C:\\Users\\dorian\\Desktop\\%s-light_intensity.png'%(s), dataset.light_intensity, vmin=0.0, vmax=1.0)
            plt.imsave('C:\\Users\\dorian\\Desktop\\%s-throughput.png'%(s), dataset.throughput, vmin=0.0, vmax=1.0)
            plt.imsave('C:\\Users\\dorian\\Desktop\\%s-bounce_type.png'%(s), dataset.bounce_type)
            plt.imsave('C:\\Users\\dorian\\Desktop\\%s-roughness.png'%(s), dataset.roughness)
    
    def test_preprocess():
        dataset = DenoiseDataset('D:\\LLPM', 8, mode='test', base_model='kpcn', use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=False, sampling='grid')
        start = time.time()
        dataset._offline_preprocess(sbmc=False, llpm=True, kpcn=True, overwrite=False)
        print(time.time() - start)

    # save_all_buffers()

    """
    out = dataset.__getitem__(0)
    print('# of patches', len(dataset.samples))
    if type(out) == tuple:
        for o in out:
            print(o.keys())
            for k in o:
                if type(o[k]) == np.ndarray or type(o[k]) == np.memmap:
                    print(k, type(o[k]), o[k].shape)
    else:
        print(out.keys())
        for k in out:
            if type(out[k]) == np.ndarray or type(out[k]) == np.memmap:
                print(k, type(out[k]), out[k].shape)
    
    if (dataset.base_model == 'sbmc'):
        plt.title('target_image')
        plt.imshow(out['target_image'])
        plt.show()
        plt.title('radiance')
        plt.imshow(out['radiance'].mean(2))
        plt.show()
        plt.title('total_r')
        plt.imshow((np.exp(10 * out['features'][...,0:3])-1).mean(2))
        plt.show()
        plt.title('specular_r')
        plt.imshow(np.exp(10 * out['features'].mean(2)[...,3:6])-1)
        plt.show()
        plt.title('subpixel')
        plt.imshow(out['features'].mean(2)[...,6])
        plt.show()
        plt.title('subpixel')
        plt.imshow(out['features'].mean(2)[...,7])
        plt.show()
        plt.title('albedo_first_r')
        plt.imshow(out['features'].mean(2)[...,8:11])
        plt.show()
        plt.title('albedo_r')
        plt.imshow(out['features'].mean(2)[...,11:14])
        plt.show()
        plt.title('normal_first_x')
        plt.imshow(out['features'].mean(2)[...,14:17])
        plt.show()
        plt.title('normal_x')
        plt.imshow(out['features'].mean(2)[...,17:20])
        plt.show()
        plt.title('depth_first')
        plt.imshow(out['features'].mean(2)[...,20])
        plt.show()
        plt.title('depth')
        plt.imshow(out['features'].mean(2)[...,21])
        plt.show()
        plt.title('visibility')
        plt.imshow(out['features'].mean(2)[...,22])
        plt.show()
        plt.title('hasHit')
        plt.imshow(out['features'].mean(2)[...,23])
        plt.show()
        plt.title('l_dir_theta')
        plt.imshow(out['features'].mean(2)[...,48])
        plt.show()
    elif (dataset.base_model == 'kpcn'):
        plt.title('target')
        plt.imshow(out['target_diffuse']*out['kpcn_albedo']+np.exp(out['target_specular'])-1)
        plt.show()
        plt.title('target_diffuse')
        plt.imshow(out['target_diffuse'])
        plt.show()
        plt.title('kpcn_diffuse_buffer')
        plt.imshow(out['kpcn_diffuse_buffer'])
        plt.show()
        plt.title('kpcn_diffuse_in')
        plt.imshow(out['kpcn_diffuse_in'][...,:3])
        plt.show()
        plt.title('diffuse_v')
        plt.imshow(out['kpcn_diffuse_in'][...,3])
        plt.show()
        plt.title('diffuse_r_dx')
        plt.imshow(out['kpcn_diffuse_in'][...,4:7])
        plt.show()
        plt.title('diffuse_r_dy')
        plt.imshow(out['kpcn_diffuse_in'][...,7:10])
        plt.show()
        plt.title('normals_x')
        plt.imshow(out['kpcn_diffuse_in'][...,10:13])
        plt.show()
        plt.title('normals_v')
        plt.imshow(out['kpcn_diffuse_in'][...,13])
        plt.show()
        plt.title('normals_x_dx')
        plt.imshow(out['kpcn_diffuse_in'][...,14:17])
        plt.show()
        plt.title('normals_x_dy')
        plt.imshow(out['kpcn_diffuse_in'][...,17:20])
        plt.show()
        plt.title('albedo_r')
        plt.imshow(out['kpcn_diffuse_in'][...,24:27])
        plt.show()
        plt.title('kpcn_albedo')
        plt.imshow(out['kpcn_albedo'])
        plt.show()
        plt.title('target_specular')
        plt.imshow(out['target_specular'])
        plt.show()
        plt.title('kpcn_specular_buffer')
        plt.imshow(out['kpcn_specular_buffer'])
        plt.show()
        plt.title('kpcn_specular_in')
        plt.imshow(out['kpcn_specular_in'][...,:3])
        plt.show()
        plt.title('albedo_r')
        plt.imshow(out['kpcn_specular_in'][...,24:27])
        plt.show()
        plt.title('kpcn_albedo')
        plt.imshow(out['kpcn_albedo'])
        plt.show()
    elif (dataset.base_model == 'amcd'):
        diffuse, specular = out
        plt.title('ref')
        plt.imshow(diffuse['ref'])
        plt.show()
        plt.title('in')
        plt.imshow(diffuse['in'])
        plt.show()
        plt.title('normals_x')
        plt.imshow(diffuse['seg'][...,:3])
        plt.show()
        plt.title('depth')
        plt.imshow(diffuse['seg'][...,3])
        plt.show()
        plt.title('albedo_r')
        plt.imshow(diffuse['seg'][...,4:7])
        plt.show()
        plt.title('ref')
        plt.imshow(specular['ref'])
        plt.show()
        plt.title('in')
        plt.imshow(specular['in'])
        plt.show()
        plt.title('normals_x')
        plt.imshow(specular['seg'][...,:3])
        plt.show()
        plt.title('depth')
        plt.imshow(specular['seg'][...,3])
        plt.show()
        plt.title('albedo_r')
        plt.imshow(specular['seg'][...,4:7])
        plt.show()
        
        plt.title('path_weight')
        plt.imshow(specular['seg'][...,7])
        plt.show()
        plt.title('radiance_wo_weight_r')
        plt.imshow(diffuse['paths'].mean(2)[...,:3])
        plt.show()
        plt.title('light_intensity_r')
        plt.imshow(diffuse['paths'].mean(2)[...,3:6])
        plt.show()
        plt.title('roughnesses')
        plt.imshow(diffuse['paths'].mean(2)[...,30])
        plt.show()
    
    import lz4.frame
    cond = 3
    start = time.time()
    if (cond == 0):
        print('Numpy read...')
        img_np = np.load('D:\\LLPM\\train\\input\\living-room-3_65.npy')
        img_np = np.log(1 + img_np) / 10.0
        #tensor = torch.from_numpy(img_np[...,:8,:]).cuda(0)
        #print(tensor.shape)
        #print(tensor.mean())
    elif (cond == 1):
        print('Binary read...')
        f = open('D:\\LLPM\\train\\input\\living-room-3_65_kpcn.data', 'rb')
        img_buf = f.read(288358400) # f.read(5452595200)
        f.close()
        img_np = np.frombuffer(img_buf, dtype=np.float32).reshape(1280,1280,-1) # ,8
    elif (cond == 2):
        print('Compressed binary read...')
        f = open('D:\\LLPM\\train\\input\\living-room-3_65_kpcn.lz4', 'rb')
        img_lz4 = f.read(214075752) # f.read(2887816237)
        f.close()
        img_buf = lz4.frame.decompress(img_lz4)
        img_np = np.frombuffer(img_buf, dtype=np.float32).reshape(1280,1280,-1) # ,8
    else:
        print('Numpy read with memmap...')
        img_np = np.load('D:\\LLPM\\train\\input\\living-room-3_65.npy', mmap_mode='r')
        img_np = np.log(1 + img_np) / 10.0
        #tensor = torch.from_numpy(img_np[...,:8,:]).cuda(0)
        #print(tensor.shape)
        #print(tensor.mean())
    print('Shape: ', img_np.shape)
    #print('First elem: ', img_np[0,0,0,0])
    #print(time.time() - start)
    #start = time.time()
    #print('Elem squaring: ', (img_np**2)[0,0,0,0])
    #print(time.time() - start)
    #start = time.time()
    print('Elem mean: ', img_np.mean())
    print(time.time() - start)
    print('Done.')
    """