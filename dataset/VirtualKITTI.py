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


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string
def read_vkitti_png_flow(flow_fn):
    # read png to bgr in 16 bit unsigned short

    bgr = cv2.imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    assert bgr.dtype == np.uint16 and _c == 3
    # b == invalid flow flag == 0 for sky or other invalid flow
    invalid = bgr[..., 0] == 0
    # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 – 1]
    out_flow = 2.0 / (2**16 - 1.0) * bgr[..., 2:0:-1].astype(np.float32) - 1
    out_flow[..., 0] *= w - 1
    out_flow[..., 1] *= h - 1
    out_flow[invalid] = 0 # or another value (e.g., np.nan)
    return out_flow, invalid
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

class VirtualKITTIDataset:
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
        self.root_path = '/share_chairilg/data/VirtualKITTI/VirtualKITTI/'
        self.data = {'image0': [], 'flow': [], 'backward': []}
        self.H = 1242
        self.W = 375
        self.read_data()
        #self.compute_occlusions()

    def read_data(self):
        image0_path = self.root_path + 'rgb/'
        flow_path = self.root_path + 'forward_flow/'
        backward_path = self.root_path + 'backward_flow/'
        im0_list = []
        flow_list = []
        backward_list = []
        clone_path = 'clone/frames/'
        for scene in os.listdir(flow_path):

            image_scene_path = os.path.join(image0_path+scene, clone_path,'rgb/Camera_0/')
            flow_scene_path = os.path.join(flow_path+scene, clone_path, 'forwardFlow/Camera_0/')
            backward_scene_path = os.path.join(backward_path+scene, clone_path, 'backwardFlow/Camera_0/')
            for file in os.listdir(image_scene_path):
                img_0 = image_scene_path + file
                im0_list.append(img_0)
                flow = flow_scene_path + file.replace('rgb', 'flow').replace('jpg', 'png')
                if os.path.exists(flow):
                    flow_list.append(flow)
                backward = backward_scene_path + file.replace('rgb', 'flow').replace('jpg', 'png').replace('flow_', 'backwardFlow_')
                if os.path.exists(backward):
                    backward_list.append(backward)

        self.data['image0'] = sorted(im0_list)
        self.data['flow'] = sorted(flow_list)
        self.data['backward'] = sorted(backward_list)

    def __len__(self):
        return len(self.data['flow'])

    def __getitem__(self, idx, mask_top=False):
        im0 = cv2.imread(self.data['image0'][idx])
        im0 = cv2.cvtColor(im0,cv2.COLOR_BGR2RGB )
        flow, invalid_flow = read_vkitti_png_flow(self.data['flow'][idx])
        occlusions = cv2.imread(self.data['flow'][idx].replace('forward_flow', 'occlusions'))
        occlusions = occlusions[...,0:1] > 0
        #flow = (full_flow[...,1:3].astype(np.float32) - 2**15) / 64.0
        #occ_flow = (full_flow[...,0] != 0).astype(np.float32)
        if self.transform:
            im0 = self.transform(np.array(im0[:,:,:3]))
            flow = torch.Tensor(np.nan_to_num(flow[:,:,:2])).permute(2, 0, 1)
            invalid_flow = torch.Tensor(invalid_flow)
        c, h, w = im0.shape
        mask = (torch.FloatTensor(1, h, w).uniform_() > self.density).float()
        occlusions = torch.tensor(occlusions).float().permute(2,0,1)
        m_flow = torch.zeros_like(flow)
        if mask_top:
            mask = (torch.FloatTensor(1, h, w).uniform_() > 0.7).float()
            percentage = np.random.randint(low=25,high=50) / 100
            lines_to_mask = int(h * percentage)
            mask[:,0:lines_to_mask,:] = 0.0
        borders = torch.zeros_like(occlusions)
        borders[:,15:-15,15:-15] = 1.
        mask *= 1-(occlusions * borders)
        indices = torch.cat((mask,mask), dim=0).bool()

        m_flow[indices] = flow[indices]

        return im0, im0, mask, flow, m_flow, 1.0 - invalid_flow.unsqueeze(0).float()

    def compute_occlusions(self):
        y_coords, x_coords = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
        index_map = torch.stack((x_coords, y_coords), dim=0).unsqueeze(0).float()
        for image, forward, backward in zip(self.data['image0'],self.data['flow'], self.data['backward']):
            forward_flow = torch.tensor(read_vkitti_png_flow(forward)[0]).permute(2,0,1).unsqueeze(0)
            #forward_flow = torch.tensor((forward[...,1:3].astype(np.float32) - 2**15) / 64.0).permute(2,0,1).unsqueeze(0)
            backward_flow = torch.tensor(read_vkitti_png_flow(backward)[0]).permute(2,0,1).unsqueeze(0)
            #backward_flow = torch.tensor((backward[...,1:3].astype(np.float32) - 2**15) / 64.0).permute(2,0,1).unsqueeze(0)
            im1 = backward.replace('backward_flow','rgb').replace('backwardFlow', 'rgb').replace('png', 'jpg')
            im0 = cv2.imread(image)
            image_tensor = torch.tensor(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float()
            im1 = cv2.imread(im1)
            image_tensor2 = torch.tensor(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).float()
            # Compute the flow difference
            """flow_diff = forward_flow - backward_flow

            # Create grids for forward and backward flow
            h, w = flow_diff.size(2), flow_diff.size(3)
            grid_forward = torch.stack(torch.meshgrid(torch.arange(0, h), torch.arange(0, w)), dim=2).float().to(
                forward_flow.device)
            grid_forward = grid_forward + forward_flow.permute(0, 2, 3, 1)  # Add flow to grid

            grid_backward = torch.stack(torch.meshgrid(torch.arange(0, h), torch.arange(0, w)), dim=2).float().to(
                backward_flow.device)
            grid_backward = (grid_backward + backward_flow.permute(0, 2, 3, 1))  # Add flow to grid

            H = h * torch.ones_like(forward_flow)
            W = w * torch.ones_like(forward_flow)
            hw = torch.stack((H[:,0,:,:], W[:,0,:,:]), dim=-1)

            grid_forward /= hw
            grid_backward /= hw
            # Use grid_sample to sample pixels
            sampled_forward = F.grid_sample(image_tensor, grid_forward)
            sampled_backward = F.grid_sample(image_tensor, grid_backward)
            # Compute occlusion map by comparing the difference between sampled pixels
            occlusion_map = torch.abs(sampled_forward - sampled_backward).sum(dim=1) > 0
            occlusion_map = torch.cat([occlusion_map]*3, dim=0)"""
            W,H = 1242, 375
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)

            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)

            xx = xx.view(1, 1, H, W).repeat(1, 1, 1, 1)

            yy = yy.view(1, 1, H, W).repeat(1, 1, 1, 1)

            grid = torch.cat((xx, yy), 1).float()

            vgrid = grid + forward_flow

            vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0

            vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

            warped_image = torch.nn.functional.grid_sample(grid, vgrid.permute(0, 2, 3, 1), 'nearest')

            vgrid = grid + backward_flow

            vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0

            vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
            warped_image = torch.nn.functional.grid_sample(warped_image, vgrid.permute(0, 2, 3, 1), 'nearest')
            """vgrid = grid + backward_flow

            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0

            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

            #warped_image = torch.nn.functional.grid_sample(warped_image, vgrid.permute(0, 2, 3, 1), 'nearest')"""
            occlusion_map = torch.abs(warped_image - grid).mean(dim=1) > 5
            occlusion_map = torch.stack([occlusion_map]*3, dim=1).float()
            occ_path = forward.replace('forward_flow', 'occlusions')

            os.makedirs(occ_path[:occ_path.rindex('/')+1], exist_ok=True)
            plt.imsave(occ_path,occlusion_map.squeeze().permute(1,2,0).numpy())
            #plt.imshow((((255*occlusion_map))).squeeze().permute(1,2,0).numpy().astype(np.uint8))
            #plt.show()
    def check_validity(self, mask):
        kernel = torch.ones(1,1,11,11)
        mask_p = F.pad(mask, (5,5,5,5), mode='reflect')
        convolved = F.conv2d(mask_p, kernel)

        confidence = (convolved > 5).float()
        """plt.imshow(confidence.squeeze().cpu().numpy())
        plt.show()"""
        return confidence