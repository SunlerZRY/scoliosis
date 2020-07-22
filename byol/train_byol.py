import os
import sys
import argparse
import math

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, datasets
import numpy as np
from collections import defaultdict

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('/home/hdc/guke/')  # noqa

from byol import BYOL  # noqa
import moco.moco.loader as loader  # noqa
from moco.moco.dataset_guke import DegreesData  # noqa
from moco.moco.utils import NonLinearColorJitter  # noqa


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def cleanup():
    dist.destroy_process_group()


def main(gpu, args):
    ckpt_path_save = os.path.join(
        args.ckpt_path_save, args.experiment_name + "_" + str(args.experiment_id))
    if not os.path.exists(ckpt_path_save):
        os.makedirs(ckpt_path_save)
    print("ckpt_path_save:", ckpt_path_save)

    rank = args.nr * args.gpus + gpu
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    normalize = transforms.Normalize(mean=[0.4771, 0.4769, 0.4355],
                                     std=[0.2189, 0.1199, 0.1717])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomCrop(args.image_size),
        NonLinearColorJitter(),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        # ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(),
        normalize
    ]
    transform = loader.TwoCropsTransform(transforms.Compose(augmentation))
    # dataset
    traindir = [
        "/data/gukedata/train_data/0-10",
        "/data/gukedata/train_data/11-15",
        "/data/gukedata/train_data/16-20",
        "/data/gukedata/train_data/21-25",
        "/data/gukedata/train_data/26-45",
        "/data/gukedata/train_data/46-",
        "/data/gukedata/test_data/0-10",
        "/data/gukedata/test_data/11-15",
        "/data/gukedata/test_data/16-20",
        "/data/gukedata/test_data/21-25",
        "/data/gukedata/test_data/26-45",
        "/data/gukedata/test_data/46-"
    ]

    train_dataset = DegreesData(traindir, transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    # model
    model = models.wide_resnet50_2()

    model = BYOL(model, image_size=args.image_size, hidden_layer="avgpool")
    model = model.cuda(gpu)

    # distributed data parallel
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    log_path = os.path.join(
        args.log_path, args.experiment_name + "_" + str(args.experiment_id))
    # TensorBoard writer

    if gpu == 0:
        writer = SummaryWriter(log_path)

    # solver
    global_step = 0
    for epoch in range(args.num_epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)
        metrics = defaultdict(list)
        for step, ((x_i, x_j), _) in enumerate(train_loader):
            x_i = x_i.cuda(non_blocking=True)
            x_j = x_j.cuda(non_blocking=True)

            loss = model(x_i, x_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.module.update_moving_average()  # update moving average of target encoder

            if step % 1 == 0 and gpu == 0:
                print(
                    f"Step [{step}/{len(train_loader)}]:\tLoss: {loss.item()}")

            if gpu == 0:
                writer.add_scalar("Loss/train_step", loss, global_step)
                metrics["Loss/train"].append(loss.item())
                global_step += 1

        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            if gpu == 0:
                print("Epoch:", epoch, 'Learning_rate:', lr)
                writer.add_scalar(
                    'Learning_rate', lr, epoch)
            break
        if gpu == 0:
            # write metrics to TensorBoard
            for k, v in metrics.items():
                writer.add_scalar(k, np.array(v).mean(), epoch)

            if epoch % args.checkpoint_epochs == 0:
                if gpu == 0:
                    print(f"Saving model at epoch {epoch}")
                    torch.save(model.state_dict(),
                               ckpt_path_save + "/model-{epoch}.pt")

                # let other workers wait until model is finished
                # dist.barrier()

    # save your improved network
    if gpu == 0:
        torch.save(model.state_dict(), ckpt_path_save + "/model-final.pt")

    cleanup()


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.num_epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.2 if epoch >= milestone else 1.

    optimizer.param_groups[0]['lr'] = lr
    # for param_group in optimizer.param_groups:
    #     print("param_group lr: ", param_group['lr'])

    return lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path_save', '-ckpt_s',
                        default='/data/gukedata/ckpt/', help='checkpoint path to save')
    parser.add_argument('--log_path', '-lp',
                        default='/data/gukedata/log_sl/', help='log path')
    parser.add_argument('--experiment_id', '-eid',
                        default='0', help='experiment id')
    parser.add_argument('--experiment_name', '-name',
                        default='byol_wide_resnet50_2', help='experiment name')
    parser.add_argument("--image_size", default=224,
                        type=int, help="Image size")
    parser.add_argument(
        "--learning_rate", default=0.1, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", default=1000, type=int, help="Number of epochs to train for."
    )
    parser.add_argument('-a', '--arch', metavar='ARCH', default='wide_resnet50_2',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument(
        "--checkpoint_epochs",
        default=5,
        type=int,
        help="Number of epochs between checkpoints/summaries.",
    )
    parser.add_argument(
        "--num_workers",
        default=16,
        type=int,
        help="Number of data loading workers (caution with nodes!)",
    )
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')
    parser.add_argument('--schedule', default=[60, 100, 120], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes",
    )
    parser.add_argument("--gpus", default=4, type=int,
                        help="number of gpus per node")
    parser.add_argument("--nr", default=0, type=int,
                        help="ranking within the nodes")
    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "10001"
    args.world_size = args.gpus * args.nodes

    # Initialize the process and join up with the other processes.
    # This is “blocking,” meaning that no process will continue until all processes have joined.
    mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
