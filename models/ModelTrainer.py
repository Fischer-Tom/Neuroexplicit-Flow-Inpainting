from os.path import join

import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import flow_vis
from imagelib.core import inverse_normalize
from imagelib.inout import write_flo_file
from tqdm import tqdm
from utils.loss_functions import smoothness_loss

from flow_vis import flow_to_color

class ModelTrainer:
    def __init__(self, net, optimizer, gpu, train_iter, **kwargs):
        self.net = net
        self.train_iters = 0
        self.total_iters = train_iter
        self.gpu = gpu
        self.optimizer = self.get_optimizer(
            optimizer, kwargs["lr"], kwargs["weight_decay"]
        )
        self.scheduler = self.net.get_scheduler(self.optimizer)

    def get_optimizer(self, type, lr, weight_decay):
        if type == "Adam":
            return torch.optim.Adam(
                self.net.parameters(), lr=lr, weight_decay=weight_decay
            )
        raise ValueError("Invalid Optimizer. Choices are: Adam")

    def load_parameters(self, path, **kwargs):
        dict = torch.load(path)
        if kwargs["mode"] == "depth":
            new_dict = {k: v for k, v in dict.items() if "grad" not in k}
            self.net.load_state_dict(new_dict, strict=False)
        else:
            self.net.load_state_dict(dict)

    def save_parameters(self, path):
        torch.save(self.net.state_dict(), join(path, f"model{self.train_iters}.pt"))
        torch.save(
            self.optimizer.state_dict(), join(path, f"optimizer{self.train_iters}.pt")
        )
        torch.save(
            self.scheduler.state_dict(), join(path, f"scheduler{self.train_iters}.pt")
        )

    def train(self, loader):
        self.net.train()

        running_loss = 0.0
        iterations = 0
        I1, Mask, Flow, predict_flow = None, None, None, None
        with tqdm(loader, unit="batch") as tepoch:
            for i, sample in enumerate(tepoch):
                if i % 100 == 0:
                    running_loss = 0.0
                    iterations = 0
                I1 = sample[0].cuda()
                Mask = sample[2].cuda()
                Flow = sample[3].cuda()
                Masked_Flow = sample[4].cuda()

                # Time Iteration duration
                self.optimizer.zero_grad(set_to_none=True)
                # Query Model
                predict_flow = self.net(I1, Mask, Masked_Flow)

                # LossFlow = Flow.clone()

                batch_risk = self.net.get_loss(
                    predict_flow, Flow, Mask, train=True
                )


                # Update Weights and learning rate
                batch_risk.backward()
                self.optimizer.step()
                self.net.update_lr(self.scheduler, self.train_iters)

                with torch.no_grad():
                    self.net.constrain_weight()
                # Update running loss
                running_loss += batch_risk.item()
                iterations += 1
                self.train_iters += 1
                if self.train_iters > self.total_iters:
                    break

                tepoch.set_postfix(
                    loss=running_loss / iterations
                )

        Flow_vis = flow_to_color(Flow[0].detach().cpu().permute(1, 2, 0).numpy())
        Pred_vis = flow_to_color(predict_flow[0].detach().cpu().permute(1, 2, 0).numpy())
        I1_vis = inverse_normalize(I1[0].cpu())
        Mask_vis = torch.cat((Mask[0], Mask[0], Mask[0]), dim=0).detach().cpu()
        images = torch.stack(
            (
                I1_vis,
                torch.tensor(Flow_vis).permute(2, 0, 1),
                torch.tensor(Mask_vis),
                torch.tensor(Pred_vis).permute(2, 0, 1),
            )
        )

        return running_loss / iterations, images

    def validate(self, loader):
        self.net.eval()
        running_loss = 0.0
        iterations = 0
        I1, Mask, Flow, predict_flow = None, None, None, None
        with torch.no_grad():
            with tqdm(enumerate(loader), unit="batch") as tqiter:
                for i, sample in tqiter:
                    sample = [samp.cuda(self.gpu) for samp in sample]

                    I1, I2 = sample[0:2]
                    Mask = sample[2]
                    Flow = sample[3]
                    Masked_Flow = sample[4]
                    # Query Model

                    predict_flow = self.net(I1, Mask, Masked_Flow)

                    batch_risk = self.net.get_loss(
                        predict_flow, Flow, Mask
                    )
                    # Update running loss

                    running_loss += batch_risk.item()

                    iterations += 1
                    tqiter.set_postfix(
                        loss=(running_loss) / iterations,
                    )

        Flow_vis = flow_to_color(Flow[0].detach().cpu().permute(1, 2, 0).numpy())
        Pred_vis = predict_flow[0].detach().cpu().permute(1, 2, 0).numpy()
        I1_vis = inverse_normalize(I1[0].cpu())
        Mask_vis = torch.cat((Mask[0], Mask[0], Mask[0]), dim=0).detach().cpu()
        images = torch.stack(
            (
                I1_vis,
                torch.tensor(Flow_vis).permute(2, 0, 1),
                torch.tensor(Mask_vis).permute(2, 0, 1),
                torch.tensor(Pred_vis).permute(2, 0, 1),
            )
        )

        return running_loss / iterations, images

