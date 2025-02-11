from os.path import join

import matplotlib.pyplot as plt
import torch
import torch.optim
import flow_vis
import torch.nn.functional as F
from imagelib.core import inverse_normalize
from utils.loss_functions import EPE_Loss
from tqdm import tqdm
import numpy as np
from imagelib.inout import write_flo_file
class GANModelTrainer:


    def __init__(self, G,C, optimizer, gpu, train_iter, **kwargs):

        self.G = G
        self.C = C
        self.train_iters = 0
        self.total_iters = train_iter
        self.gpu = gpu
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=5e-5,betas=(0.5,0.999),weight_decay=4e-4)
        self.optimizer_C = torch.optim.Adam(self.C.parameters(), lr=5e-5,betas=(0.5,0.999),weight_decay=4e-4)
        self.lambda_GP = 10.
        self.critic_iters = 1


    def get_optimizer(self, type, lr, weight_decay):

        if type == "Adam":
            return torch.optim.Adam(self.net.parameters(), lr = lr, weight_decay=weight_decay)
        raise ValueError("Invalid Optimizer. Choices are: Adam")



    def load_parameters(self, path, **kwargs):
        self.G.load_state_dict(torch.load(path))

    def save_parameters(self, path):
        torch.save(self.G.state_dict(), join(path, f"model{self.train_iters}.pt"))



    def train(self, loader):
        self.G.train()
        self.C.train()

        running_loss = 0.0
        iterations = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        I1, Mask, Flow, predict_flow = None, None, None, None
        with tqdm(loader, unit="batch") as tepoch:

            for i, sample in enumerate(tepoch):
                tepoch.set_description(f"Iterations {i}")
                sample = [samp.cuda(self.gpu) for samp in sample]

                I1, I2 = sample[0:2]
                Mask = sample[2]
                real= sample[3] / 100.0
                Masked_Flow = sample[-1] / 100.0

                # Time Iteration duration
                #indices = torch.cat((Mask,Mask), dim=1)
                start.record()
                r = (1 - Mask) * torch.randn_like(Masked_Flow)
                with torch.no_grad():
                    self.G.eval()
                    MaskSave = torch.cat((Mask,Mask,Mask),dim=1)
                    plt.imsave(f'./Mask-{i // 2000}.png', MaskSave[0].cpu().permute(1, 2, 0).numpy())
                    plt.imsave(f'./flow-{i // 2000}.png', flow_vis.flow_to_color(real[0].cpu().permute(1, 2, 0).numpy()))
                    plt.imsave(f'./image-{i // 2000}.png', inverse_normalize(I1[0].cpu()).permute(1, 2, 0).numpy())
                    self.G.load_state_dict(torch.load("./checkpoints/WGAIN/WGAIN_1.pt"))
                    images = self.G(I1,Mask,Masked_Flow,r)
                    images = (1 - Mask) * images + Mask * Masked_Flow

                    images = images * 100.0  # 1353.2810

                    plt.imsave(f'./sample-{i // 2000}.png',
                               flow_vis.flow_to_color(images[0].cpu().permute(1, 2, 0).numpy()))

                exit()

                for _ in range(self.critic_iters):
                    fake = self.G(I1, Mask, Masked_Flow,r)
                    # Query Model
                    fake_guess = self.C(I1,fake,Mask).reshape(-1)
                    real_guess = self.C(I1,real,Mask).reshape(-1)
                    gp = self.get_gradient_penalty(I1,real, fake, Mask)
                    loss_C = -(torch.mean(real_guess) - torch.mean(fake_guess)) + self.lambda_GP * gp

                    # Update Weights and learning rate
                    self.optimizer_C.zero_grad()
                    loss_C.backward(retain_graph=True)
                    self.optimizer_C.step()
                    #torch.nn.utils.clip_grad_norm_(self.C.parameters(),1.0)

                fake_guess = self.C(I1,fake,Mask).reshape(-1)
                mae = EPE_Loss(100*fake,100*real)#torch.mean(torch.abs(fake-real))
                loss_gen = -torch.mean(fake_guess) + mae
                self.optimizer_G.zero_grad()
                loss_gen.backward()
                self.optimizer_G.step()


                end.record()
                torch.cuda.synchronize()
                # Update running loss
                fake = (1-Mask)*fake + Mask*Masked_Flow
                running_loss += EPE_Loss(100.0*real, 100.0*fake).item()
                iterations += 1
                self.train_iters += 1
                tepoch.set_postfix(critic_loss=-loss_C.item(), loss=running_loss / iterations)
                if self.train_iters > self.total_iters:
                    break
                """
                if not (i % 50):
                    plt.imsave(f'./sample-{i // 2000}.png',
                               flow_vis.flow_to_color(100.0*fake[0].detach().cpu().permute(1, 2, 0).numpy()))
                    plt.imsave(f'./real-{i // 2000}.png',
                               flow_vis.flow_to_color(100.0*real[0].detach().cpu().permute(1, 2, 0).numpy()))
                    #print(running_loss / iterations)
                """

        Flow_vis = flow_vis.flow_to_color(100.0*real[0].detach().cpu().permute(1,2,0).numpy())
        Pred_vis = flow_vis.flow_to_color(100.0*fake[0].detach().cpu().permute(1, 2, 0).numpy())
        I1_vis = inverse_normalize(I1[0].detach().cpu())
        Masked_vis = flow_vis.flow_to_color(100.0*Masked_Flow[0].detach().cpu().permute(1, 2, 0).numpy())
        Mask_vis = torch.cat((Mask[0],Mask[0],Mask[0]),dim=0).detach().cpu()
        images = torch.stack((I1_vis,Mask_vis,torch.tensor(Flow_vis).permute(2,0,1),torch.tensor(Masked_vis).permute(2,0,1),torch.tensor(Pred_vis).permute(2,0,1)))

        self.critic_iters = 1
        return running_loss / iterations, start.elapsed_time(end), images

    def validate(self, loader, KITTI=False):
        self.G.eval()
        self.C.eval()

        running_loss = 0.0
        running_mae = 0.0
        running_FL = 0.0
        iterations = 0
        I1, Mask, Flow, predict_flow = None, None, None, None
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with tqdm(loader, unit="batch") as tepoch:
                for i,sample in enumerate(tepoch):

                    tepoch.set_description(f"Iterations {iterations}")
                    sample = [samp.cuda(self.gpu) for samp in sample]

                    I1, I2 = sample[0:2]
                    Mask = sample[2]
                    Flow = sample[3] / 100.0
                    Masked_Flow = sample[4] / 100.0
                    if KITTI:
                        _,_,w,h = I1.shape
                        pad_w = (64 - (w % 64)) // 2
                        pad_h = (64 - (h % 64)) // 2
                        uneven_w = w % 2 == 1
                        uneven_h = h % 2 == 1
                        I1 = F.pad(I1, (pad_h, pad_h + uneven_h, pad_w, pad_w + uneven_w))
                        Mask = F.pad(Mask, (pad_h, pad_h + uneven_h, pad_w, pad_w + uneven_w))
                        Masked_Flow = F.pad(Masked_Flow, (pad_h, pad_h + uneven_h, pad_w, pad_w + uneven_w))
                        occ_flow = sample[5]

                    r = (1 - Mask) * torch.randn_like(Masked_Flow)

                    # Query Model
                    start.record()

                    predict_flow = self.G(I1, Mask, Masked_Flow, r) * 100.0

                    if KITTI:
                        I1 = I1[:, :, pad_w:-pad_w - uneven_w, pad_h:-pad_h - uneven_h]
                        predict_flow = predict_flow[:,:, pad_w:-pad_w - uneven_w, pad_h:-pad_h - uneven_h]
                        Mask = Mask[:,:, pad_w:-pad_w - uneven_w, pad_h:-pad_h - uneven_h]
                        Masked_Flow = Masked_Flow[:,:, pad_w:-pad_w - uneven_w, pad_h:-pad_h - uneven_h]
                    FL, batch_risk = EPE_Loss(
                        predict_flow, Flow*100, Mask, train=False
                    )
                    end.record()
                    torch.cuda.synchronize()
                    # Update running loss

                    running_loss += batch_risk.item()

                    rmse = 0.0  # ((predict_flow - Flow)[occ_flow.bool()] **2).mean().sqrt()
                    # mae = 0.0#torch.abs(predict_flow - Flow)[occ_flow.bool()].mean()
                    running_mae += batch_risk.item()
                    running_FL += FL.item()
                    iterations += 1
                    tepoch.set_postfix(
                        loss=(running_loss) / iterations,
                        FL=running_FL / iterations,
                        mae=running_mae / iterations,
                    )

                    Flow_vis = flow_vis.flow_to_color(Flow[0].detach().cpu().permute(1, 2, 0).numpy())
                    Pred_vis = flow_vis.flow_to_color(
                        torch.nan_to_num_(predict_flow[0]).detach().cpu().permute(1, 2, 0).numpy())
                    Masked_vis = flow_vis.flow_to_color(Masked_Flow[0].detach().cpu().permute(1, 2, 0).numpy())
                    I1_vis = inverse_normalize(I1[0].detach().cpu())
                    Mask_vis = torch.cat((Mask[0], Mask[0], Mask[0]), dim=0).detach().cpu()
                    images = torch.stack((I1_vis, Mask_vis, torch.tensor(Flow_vis).permute(2, 0, 1),
                                          torch.tensor(Masked_vis).permute(2, 0, 1),
                                          torch.tensor(Pred_vis).permute(2, 0, 1)))
                    write_flo_file(f"sampleImages/Pred_WGAIN_{i}.flo",predict_flow[0].detach().cpu().permute(1,2,0).numpy())


                    I1 = images[0, ::]
                    Flow = images[2, ::]
                    Masked = images[3, ::]
                    Pred = images[4, ::]

                    plt.imsave(f"sampleImages/Image_WGAIN_{i}.png", I1.permute(1, 2, 0).numpy())
                    plt.imsave(f"sampleImages/Flow_WGAIN_{i}.png", Flow.permute(1, 2, 0).numpy().astype(np.uint8))
                    plt.imsave(f"sampleImages/Masked_WGAIN_{i}.png",
                               Masked.permute(1, 2, 0).numpy().astype(np.uint8))
                    plt.imsave(f"sampleImages/Pred_WGAIN_{i}.png", Pred.permute(1, 2, 0).numpy().astype(np.uint8))

        return running_loss / iterations , start.elapsed_time(end), images



    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def get_gradient_penalty(self,I,real_guess,fake_guess, M):
        b,c,_,_ = real_guess.shape
        eps = torch.rand((b,1,1,1), device=real_guess.device).repeat(1,c,1,1)
        difference = fake_guess - real_guess
        interpolate = real_guess + (eps*difference)
        int_score = self.C(I,interpolate,M)
        grad = torch.autograd.grad(inputs=interpolate,
                                outputs=int_score,
                                grad_outputs=torch.ones_like(int_score),
                                create_graph=True,
                                retain_graph=True, )[0]
        grad = grad.view(grad.shape[0], -1)
        grad_norm = grad.norm(2, dim=1)
        gp = torch.mean((grad_norm - 1.) ** 2)
        return gp