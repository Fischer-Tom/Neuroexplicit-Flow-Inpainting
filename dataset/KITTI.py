from os.path import join
import os

import flow_vis
import torch
from imagelib.inout import read
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from utils.flowpy import flow_to_rgb
import matplotlib.pyplot as plt
from imagelib.inout import write_flo_file
from skimage.filters import median
from skimage.morphology import square

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def write_flow(flow, filename, occ_flow=None):
    """Write optical flow in Middlebury .flo format"""
    if len(flow.shape) == 4:
        flow = flow.squeeze(0)
    if occ_flow is None:
        valid = torch.ones_like(flow)
    else:
        valid = occ_flow
    flow *= valid
    flow = 64*flow + 2**15
    flow = torch.cat((valid[0:1], flow), dim=0)
    np_flow = torch.permute(flow, (1, 2, 0)).numpy().astype(np.uint16)

    cv2.imwrite(filename, np_flow)

class KITTIDataset:
    def __init__(self, img_dir, density, presmooth, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]),
                 mode='train',
                 type='IP'
                 ):
        self.mode = mode
        self.type = type
        self.transform = transform
        self.flow_transform = None#transforms.CenterCrop((374, 1242))
        self.density = density
        self.root_path = '/share_chairilg/data/KITTI/stereo15/data_scene_flow/training/'
        self.data = {'image0': [], 'image1': [], 'flow': []}
        self.read_data()

    def read_data(self):
        image0_path = self.root_path + 'image_2/'
        flow_path = self.root_path + 'flow_occ/'
        im0_list = []
        im1_list = []
        flow_list = []
        for file in os.listdir(flow_path):
            if file.endswith('.flo'):
                continue
            img_1 = image0_path + file
            img_2 = list(img_1)
            img_2[-5] = '1'

            im0_list.append(img_1)
            im1_list.append(''.join(img_2))
            flow_list.append(flow_path + file)
        self.data['image0'] = sorted(im0_list)

        self.data['image1'] = sorted(im1_list)
        self.data['flow'] = sorted(flow_list)

    def __len__(self):
        return len(self.data['image0'])

    def __getitem__(self, idx):
        im0 = cv2.imread(self.data['image0'][idx])
        im1 = cv2.imread(self.data['image1'][idx])
        im0 = cv2.cvtColor(im0,cv2.COLOR_BGR2RGB )
        im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)
        full_flow = cv2.imread(self.data['flow'][idx], cv2.IMREAD_UNCHANGED)
        flow = (full_flow[...,1:3].astype(np.float32) - 2**15) / 64.0
        occ_flow = full_flow[...,0].astype(np.float32)
        if self.transform:
            im0 = self.transform(np.array(im0[:,:,:3]))
            im1 = self.transform(np.array(im1[:,:,:3]))
            flow = torch.Tensor(np.nan_to_num(flow[:,:,:2])).permute(2, 0, 1)
            occ_flow = torch.Tensor(occ_flow)


        mask = torch.multinomial(occ_flow.flatten(), int((1.0-self.density)*im0.shape[1]*im0.shape[2]))
        subsampled = torch.zeros_like(occ_flow).flatten()
        subsampled[mask] = 1.0
        subsampled = subsampled.reshape(occ_flow.shape)

        m_flow = torch.zeros_like(flow)
        #subsampled = occ_flow
        indices = torch.stack((subsampled,subsampled), dim=0).bool()


        m_flow[indices] = flow[indices]
        return im0, im1, subsampled.unsqueeze(0), flow, m_flow, occ_flow.unsqueeze(0)
    @torch.no_grad()
    def create_KITTI_dataset(self, net):
        for idx in range(len(self)):
            if idx > 10:
                break
            im0, _, occ_flow2, flow, m_flow, occ_flow = self.__getitem__(idx)
            _,w,h = im0.shape
            _,w2,h2 = flow.shape
            pad_w = (32 - (w%32)) // 2
            pad_h = (32 - (h%32)) // 2
            uneven_w = w % 2 == 1
            uneven_h = h % 2 == 1
            im0, flow, occ_flow = im0.unsqueeze(0), m_flow.unsqueeze(0), occ_flow2.unsqueeze(0)
            im0 = F.pad(im0, (pad_h, pad_h+uneven_h, pad_w, pad_w+uneven_w))
            occ_flow = F.pad(occ_flow, (pad_h, pad_h+uneven_h, pad_w, pad_w+uneven_w))
            flow = F.pad(flow, (pad_h, pad_h+uneven_h, pad_w, pad_w+uneven_w))
            new_flow = net(im0.cuda(), occ_flow.cuda(), flow.cuda()).squeeze(0).cpu()
            #new_flow = net(im0.cuda(), occ_flow.cuda(), flow.cuda()).squeeze(0).cpu()
            new_flow = new_flow[:,pad_w:-pad_w-uneven_w,pad_h:-pad_h-uneven_h]
            flow = flow[0,:, pad_w:-pad_w - uneven_w, pad_h:-pad_h - uneven_h]
            occ_flow = occ_flow[0,:,pad_w:-pad_w-uneven_w,pad_h:-pad_h-uneven_h]
            out_path = self.data['flow'][idx].replace('flow_occ_train','inpainted_flow_occ_train')
            #os.makedirs(out_path, exist_ok=True)
            mask = self.check_validity(occ_flow)
            new_flow = new_flow# * mask.squeeze(0)
            #c_1 = denoise_bilateral(new_flow[0,...].cpu().numpy(), sigma_color=0.005)
            #c_2 = denoise_bilateral(new_flow[1,...].cpu().numpy(), sigma_color=0.005)
            """c_1 = median(new_flow[0,...].cpu().numpy(), square(7), behavior='ndimage')
            c_2 = median(new_flow[1, ...].cpu().numpy(), square(7), behavior='ndimage')
            new_flow = torch.stack((torch.tensor(c_1), torch.tensor(c_2)), dim=0)"""

            #write_flow(new_flow, out_path, mask)
            out_path = out_path.replace('.png', '.flo')
            out_path = out_path.replace('inpainted_flow_occ_train', 'tmp')
            write_flo_file(out_path,new_flow.cpu().permute(1,2,0).numpy())
            write_flo_file(out_path.replace('10.flo', 'gt.flo'), flow.squeeze(0).cpu().permute(1,2,0).numpy())
            #Flow_vis = flow_vis.flow_to_color(new_flow.cpu().permute(1, 2, 0).numpy(), convert_to_bgr=True)
            #plt.imsave(out_path, 255 - Flow_vis)

    def check_validity(self, mask):
        kernel = torch.ones(1,1,11,11)
        mask_p = F.pad(mask, (5,5,5,5), mode='reflect')
        convolved = F.conv2d(mask_p, kernel)

        confidence = (convolved > 4).float()
        """plt.imshow(confidence.squeeze().cpu().numpy())
        plt.show()"""
        return confidence