from __future__ import print_function
import os
import time
import argparse
import logging
import hashlib
import copy
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torchmetrics.classification import MulticlassAccuracy
import sparselearning
from sparselearning.core import Masking, CosineDecay
from sparselearning.models import (
    AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe,
    WideResNet, MLP_CIFAR10, ResNet34, ResNet18
)
from sparselearning.utils import (
    get_mnist_dataloaders,
    get_cifar10_dataloaders,
    get_cifar100_dataloaders
)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True


# --------------------------------------------------------------------------------
# Model Registry
# --------------------------------------------------------------------------------
models = {
    'MLPCIFAR10': (MLP_CIFAR10, []),
    'lenet5': (LeNet_5_Caffe, []),
    'lenet300-100': (LeNet_300_100, []),
    'alexnet-s': (AlexNet, ['s', 10]),
    'alexnet-b': (AlexNet, ['b', 10]),
    'vgg-c': (VGG16, ['C', 10]),
    'vgg-d': (VGG16, ['D', 10]),
    'vgg-like': (VGG16, ['like', 10]),
    'wrn-28-2': (WideResNet, [28, 2, 10, 0.3]),
    'wrn-22-8': (WideResNet, [22, 8, 10, 0.3]),
    'wrn-16-8': (WideResNet, [16, 8, 10, 0.3]),
    'wrn-16-10': (WideResNet, [16, 10, 10, 0.3]),
}


def setup_logger(log_path: str):
    logger = logging.getLogger(f"main")
    logger.setLevel(logging.DEBUG) 
    # Avoid adding duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s %(name)s [rank=%(rank)s] %(levelname)s: %(message)s"
    )
    # Console handler only on rank 0 (to avoid duplicated stdout)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler((log_path))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Prevent propagation to root to avoid duplicate messages
    logger.propagate = False

    return logger


# --------------------------------------------------------------------------------
# Training / Evaluation
# --------------------------------------------------------------------------------
def train(args, model, device, train_loader, optimizer, epoch, mask, accuracy_metric, logger):
    model.train()
    total_loss = 0.0
    accuracy_metric.reset()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        if args.fp16:
            data = data.half()

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        if mask is not None:
            mask.step()
        else:
            optimizer.step()

        total_loss += loss.item() * data.size(0)
        accuracy_metric.update(output, target)

        if batch_idx % args.log_interval == 0 and args.rank == 0:
            acc = accuracy_metric.compute().item()
            logger.info(f"Train Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                        f"Loss: {loss.item():.4f}, Acc: {acc * 100:.2f}%")

    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = accuracy_metric.compute().item()
    if args.rank == 0:
        logger.info(f"Train Summary - Loss: {avg_loss:.4f}, Acc: {avg_acc * 100:.2f}%")
    return avg_acc


@torch.no_grad()
def evaluate(args, model, device, loader, accuracy_metric, logger, split="Validation"):
    model.eval()
    total_loss = 0.0
    accuracy_metric.reset()

    for data, target in loader:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        if args.fp16:
            data = data.half()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        total_loss += loss.item()
        accuracy_metric.update(output, target)

    total_loss /= len(loader.dataset)
    acc = accuracy_metric.compute().item()
    if args.rank == 0:
        logger.info(f"{split} - Loss: {total_loss:.4f}, Acc: {acc * 100:.2f}%")
    return acc


def init_distributed():
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world_size, local_rank
    
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='PyTorch Sparse Training with DDP')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--data', type=str, default='cifar10', choices=['mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--model', type=str, default='ResNet18')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--density', type=float, default=1.0)
    parser.add_argument('--valid-split', type=float, default=0.1)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--seed', type=int, default=17)
    sparselearning.core.add_sparse_args(parser)
    args = parser.parse_args()


    args.rank, world_size, local_rank = init_distributed()
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    exp_hash = hashlib.blake2b(json.dumps(vars(args), sort_keys=True).encode(), digest_size=6).hexdigest()
    args.exp_dir = f"./runs/{exp_hash}"
    if args.rank == 0:
        os.makedirs(args.exp_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.exp_dir, "log.txt"))

    if args.rank == 0:
        logger.info(f"Starting training on {world_size} GPUs with model {args.model}")

    if args.data == 'mnist':
        train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
    elif args.data == 'cifar10':
        train_loader, valid_loader, test_loader = get_cifar10_dataloaders(args, args.valid_split)
    else:
        train_loader, valid_loader, test_loader = get_cifar100_dataloaders(args, args.valid_split)

    train_loader.sampler = DistributedSampler(train_loader.dataset)
    valid_loader.sampler = DistributedSampler(valid_loader.dataset, shuffle=False)
    test_loader.sampler = DistributedSampler(test_loader.dataset, shuffle=False)


    if args.model == 'ResNet18':
        model = ResNet18(c=100 if args.data == 'cifar100' else 10)
    elif args.model == 'ResNet34':
        model = ResNet34(c=100 if args.data == 'cifar100' else 10)
    else:
        cls, cls_args = models[args.model]
        model = cls(*(cls_args + [False, False]))

    model.to(device)
    model = DDP(model, device_ids=[args.local_rank] if torch.cuda.is_available() else None)


    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 2, 3 * args.epochs // 4])

    decay = CosineDecay(args.death_rate, len(train_loader) * args.epochs)
    mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay,
                   growth_mode=args.growth, redistribution_mode=args.redistribution, args=args)
    mask.add_module(model, sparse_init=args.sparse_init, density=args.density)


    accuracy_metric = MulticlassAccuracy(num_classes=100 if args.data == 'cifar100' else 10).to(device)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        acc_train = train(args, model, device, train_loader, optimizer, epoch, mask, accuracy_metric, logger)
        acc_val = evaluate(args, model, device, valid_loader, accuracy_metric, logger, split="Validation")

        if args.rank == 0:
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                torch.save(model.state_dict(), os.path.join(args.exp_dir, "best.pt"))
                logger.info(f"New best val acc: {best_val_acc:.4f}")

        scheduler.step()

    if args.rank == 0:
        model.load_state_dict(torch.load(os.path.join(args.exp_dir, "best.pt")))
        acc_test = evaluate(args, model, device, test_loader, accuracy_metric, logger, split="Test")
        logger.info(f"Final Test Accuracy: {acc_test * 100:.2f}%")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
