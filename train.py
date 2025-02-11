#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group
import os
from models.ModelTrainer import ModelTrainer
from models.GANModelTrainer import GANModelTrainer
from models.PD_Trainer import PD_Trainer
from imagen_pytorch import (
    Unet,
    BaseUnet64,
    SRUnet256,
    SRUnet1024,
    ElucidatedImagen,
    Imagen,
)

# from imagen_pytorch import ElucidatedImagen
from torch.utils.data import Subset
import yaml
import numpy as np


parser = argparse.ArgumentParser(description="OFNet")

# Data Loader Details
parser.add_argument(
    "--batch_size", type=int, default=1, help="Number of Image pairs per batch"
)
parser.add_argument("--augment", type=bool, default=False, help="Use Data Augmentation")
parser.add_argument(
    "--seed", type=int, default=42, help="Seed for the Random Number Generator"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="dataset/Sintel",
    help="Relative or Absolute Path to the training Data",
)
parser.add_argument(
    "--dl_workers", type=int, default=4, help="Workers for the Dataloader"
)
parser.add_argument(
    "--train_split",
    type=float,
    default=0.9,
    help="Fraction of the Dataset to use for Training",
)
parser.add_argument(
    "--test_split",
    type=float,
    default=0.05,
    help="Fraction of the Dataset to use for Testing",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="FlyingThings",
    help="Dataset to use. Supports: FlyingThings, Sintel",
)

# Training Details
parser.add_argument(
    "--train_iter", type=int, default=900_000, help="Number of Epochs to train"
)
parser.add_argument(
    "--test_interval",
    type=int,
    default=1_000,
    help="After how many Epochs the model parses the Test set",
)
parser.add_argument(
    "--distributed", type=bool, default=False, help="Use Multiple GPUs for training"
)
parser.add_argument(
    "--model",
    type=str,
    default="InpaintingNet",
    help="",
)
parser.add_argument(
    "--model_mode",
    type=str,
    default="Single",
    help="Model to train. Supports: Single, GAN",
)

# Load and Save Paths
parser.add_argument("--pretrained", type=str, default="", help="Pretrained Model")
parser.add_argument(
    "--save_path", type=str, default="Train/", help="Where to save Model State"
)

# Optimization
parser.add_argument(
    "--optimizer",
    type=str,
    default="Adam",
    help="Which optimizer to use. Supports: adam",
)
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
parser.add_argument(
    "--scheduler",
    type=str,
    default="StepLR",
    help="Learning Rate Scheduler. Supports: StepLR",
)
parser.add_argument(
    "--weight_decay", type=float, default=4e-4, help="Weight decay parameter"
)
parser.add_argument(
    "--betas",
    type=tuple,
    default=(0.9, 0.999),
    help="Beta Values to use in case of Adam Optimizer",
)

# Miscellaneous
parser.add_argument(
    "--mode", type=str, default="train", help="Mode. Supports: train, test"
)
parser.add_argument("--dim", type=int, default=44, help="Model Dimension Multiplicator")
parser.add_argument("--mask", type=float, default=0.99, help="Mask Density for Sintel")

# Diffusion arguments
parser.add_argument("--tau", type=float, default=0.41, help="timestep size")
parser.add_argument(
    "--diffusion_position",
    type=str,
    default="decoder",
    help="Which diffusion type. Supports: encoder, decoder, none",
)
parser.add_argument(
    "--alpha", type=float, default=0.41, help="Free parameter for the WWW stencil"
)
parser.add_argument(
    "--grads",
    nargs="+",
    type=bool,
    default=[False, False, False, False, False, True],
    help="Which parameters to learn in dict form",
)
parser.add_argument("--lam", type=float, default=1.0, help="Diffusivity parameter")
parser.add_argument(
    "--steps",
    nargs="+",
    type=int,
    default=[5,15,30,45],
    help="How many steps per resolution",
)
parser.add_argument("--step", type=int, default=5, help="How many steps per resolution")
parser.add_argument("--disc", type=str, default="DB", help="Discretization")
parser.add_argument(
    "--learned_mode", type=int, default=5, help="How many parameters to learn"
)
parser.add_argument(
    "--subset_size", type=int, default=100, help="How many parameters to learn"
)
parser.add_argument(
    "--presmooth", type=bool, default=False, help="Gaussian Pre-Smoothing"
)


parser.add_argument(
    "--use_dt",
    type=bool,
    default=False,
    help="Whether or not we use DT in Res_InpaintingNet",
)
parser.add_argument(
    "--split_mode",
    type=str,
    default="diff",
    help="Type of diffusion. Supports: diff, id, resnet",
)
parser.add_argument(
    "--load_model", type=bool, default=False, help="If model checkpoint should be loaded"
)


def make_dict(grad_list):
    return {
        "lam": grad_list[0],
        "tau": grad_list[1],
        "conv": grad_list[2],
        "alphas": grad_list[3],
        "gamma": grad_list[4],
        "alpha": grad_list[5],
    }


args = parser.parse_args()
args.grads = make_dict(args.grads)
train_writer = SummaryWriter(args.save_path + "logs/train/" + args.model)
validation_writer = SummaryWriter(args.save_path + "logs/test/" + args.model)


def main_worker(gpu, ngpus, args):
    args.gpu = gpu
    # Load Model here
    ds = "IP"

    if args.model == "PD_Inpainting":

        unet1 = Unet(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=3,
            channels=2,
            channels_out=2,
            cond_images_channels=3,
            memory_efficient=True,
        )

        unet2 = SRUnet256(cond_images_channels=3, memory_efficient=True)

        net = ElucidatedImagen(
            unets=(unet1, unet2),
            image_sizes=(96, 384),
            random_crop_sizes=(None, None),
            cond_drop_prob=0.0,
            num_sample_steps=(48, 48),
            sigma_min=0.002,
            sigma_max=(120, 480),
            sigma_data=1,
            rho=7,
            P_mean=-1.2,
            P_std=1.2,
            S_churn=80,
            S_tmin=0.05,
            S_tmax=50,
            S_noise=1.003,
            condition_on_text=False,
            channels=2,
            auto_normalize_img=False,
        )

    else:

        if "InpaintingNet" in args.model:
            from models.InpaintingNet import InpaintingNetwork as model
        elif "FlowNetS+" in args.model:
            from models.FlowNetSP import FlowNetSP as model
        elif "WGAIN" in args.model:
            from models.WGAIN import WGAIN as model
        else:
            raise ImportError()


        net = model(**vars(args))
    if args.distributed:
        init_process_group(backend="nccl", world_size=ngpus, rank=args.gpu)
        torch.cuda.set_device(args.gpu)
        net.cuda(args.gpu)

        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.gpu], find_unused_parameters=True
        )

        print(f"Model has been loaded on GPU {args.gpu}")

    else:
        net = net.cuda(args.gpu)
    # Datasets and Loaders
    try:
        if args.dataset == "Sintel":
            from dataset.Sintel import SintelDataset
            from torchvision import transforms

            dataset = SintelDataset
            params = {"batch_size": 4, "shuffle": True, "num_workers": 8}
            # Datasets and Loaders
            val_dataset = dataset(args.data_path, args.mask, mode="test")
            train_dataset = dataset(args.data_path, args.mask, mode="train")
            train_loader = torch.utils.data.DataLoader(train_dataset, **params)
            validation_loader = torch.utils.data.DataLoader(val_dataset, **params)
        elif args.dataset == "FlyingThings":
            torch.manual_seed(51)
            from dataset.FlyingThings import FlyingThingsDataset
            from dataset.Sintel import SintelDataset

            from torchvision import transforms

            dataset = FlyingThingsDataset
            sintel_dataset = SintelDataset
            params = {"batch_size": 32, "shuffle": True, "num_workers": 8}
            # Datasets and Loaders
            val_dataset = dataset(
                args.data_path,
                args.mask,
                mode="test",
                type=ds,
                presmooth=args.presmooth,
            )
            sintel_val_dataset = sintel_dataset(
                args.data_path,
                args.mask,
                mode="test",
                type=ds
            )
            train_dataset = dataset(
                args.data_path,
                args.mask,
                mode="train",
                type=ds,
                presmooth=args.presmooth,
            )
            if args.subset_size != 100:
                samples = int(train_dataset.__len__() * (args.subset_size / 100))
                train_dataset = Subset(train_dataset, torch.arange(samples))
            train_loader = torch.utils.data.DataLoader(train_dataset, **params)
            val_dataset = Subset(
                val_dataset, torch.randint(val_dataset.__len__(), (512,))
            )
            validation_loader = torch.utils.data.DataLoader(val_dataset, **params)
            sintel_validation_loader = torch.utils.data.DataLoader(
                sintel_val_dataset, **params
            )
        else:
            raise ImportError()
    except ImportError:
        print("Invalid Dataset Choice. Supported are: FlyingGeometry, FlyingChairs")
        exit(1)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f"Created Model {args.model} with {pytorch_total_params} total Parameters")
    # Load ModelTrainer and Potentialy saved state
    trainer = None
    if args.model_mode == "Single":
        trainer = ModelTrainer(net, **vars(args))
    elif args.model_mode == "PD":
        trainer = PD_Trainer(net, **vars(args))
    else:
        trainer = GANModelTrainer(net.G, net.C, **vars(args))
    if args.load_model:
        checkpoints = yaml.safe_load(open("./checkpoints.yaml", "r"))
        density = 1 - args.mask
        # density = 0.05
        path = checkpoints["models"][str(args.model)][int(round(100 * density))]
        trainer.load_parameters(path, mode="flow")
    if args.mode == "test":
        # test_dataset = dataset(os.path.join(args.data_path, f'test'))
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
        #                                          shuffle=True, num_workers=args.dl_workers)

        # train_dataset.create_KITTI_dataset(trainer.net)
        # exit()
        test_risk, inf_speed, images = trainer.validate(validation_loader, False)
        print(
            f"Mask Density: {args.mask}, FlyingThings Test Risk is {test_risk:.5f} with inference time {inf_speed:.3f}"
        )

        I1 = images[0, ::]
        Flow = images[2, ::]
        Masked = images[3, ::]
        Pred = images[4, ::]
        plt.imsave(f"sampleImages/Image_KITTI.png", I1.permute(1, 2, 0).numpy())
        plt.imsave(
            f"sampleImages/Flow_KITTI.png",
            Flow.permute(1, 2, 0).numpy().astype(np.uint8),
        )
        plt.imsave(
            f"sampleImages/Masked_KITTI.png",
            Masked.permute(1, 2, 0).numpy().astype(np.uint8),
        )
        plt.imsave(
            f"sampleImages/Pred_KITTI.png",
            Pred.permute(1, 2, 0).numpy().astype(np.uint8),
        )

        # test_risk, inf_speed, samples = trainer.validate(sintel_validation_loader)
        # print(f"Mask Density: {args.mask}, Sintel Test Risk is {test_risk:.5f} with inference time {inf_speed:.3f}")

        return

    test_epochs = 1
    while trainer.train_iters < args.train_iter:
        risk, samples = trainer.train(
            train_loader
        )
        if args.gpu == 0:
            train_writer.add_scalar("Train Risk", risk, trainer.train_iters)
            print(
                f"[Training Iterations|Risk | Train time]: {trainer.train_iters} | {risk:.5f}"
            )
            train_writer.add_images("Train Set Samples", samples, trainer.train_iters)

            if trainer.train_iters > test_epochs * args.test_interval:
                validation_risk, samples = trainer.validate(
                    validation_loader
                )
                validation_writer.add_scalar("Test Risk", validation_risk, test_epochs)
                validation_writer.add_images(
                    "Validation Set Samples", samples, test_epochs
                )

                print(
                    f"[Test Epochs| Test Risk | Inference Time]: {test_epochs} | {validation_risk:.5f} "
                )

                trainer.save_parameters(args.save_path + f"checkpoints/{args.model}")
                test_epochs += 1


def main():
    os.makedirs(args.save_path + f"checkpoints/{args.model}/", exist_ok=True)
    os.makedirs(args.save_path + "logs", exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "logs", "train"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "logs", "test"), exist_ok=True)
    n_gpus = torch.cuda.device_count()

    if args.distributed:
        torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)


if __name__ == "__main__":
    main()
