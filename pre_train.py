from argparse import Namespace
import torch
from get_dataset import GetTransformedDataset
from resnet_impl import get_resnet18
from simclr_framework import simclr_framework

def main():
    args = Namespace
    args.batch_size = 500
    args.device = torch.device('cuda')
    args.disable_cuda = False
    args.epochs = 20
    args.fp16_precision = False
    args.gpu_index = -1
    args.log_every_n_steps = 1
    args.lr = 3e-4
    args.n_views = 2
    args.out_dim = 128
    args.seed = 1
    args.temperature = 0.07
    args.weight_decay = 0.0008
    args.workers = 4

    
    dataset = GetTransformedDataset()
    print('dataset loaded..')
    train_dataset = dataset.get_cifar10_train(args.n_views)
    print('train_dataset loaded..')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    print('train_loader loaded..')
    model = get_resnet18(out_dim=args.out_dim)
    print('model loaded..')
    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, weight_decay=args.weight_decay)
    print('optimizer loaded..')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                        last_epoch=-1)
    print('scheduler loaded..')
    simclr = simclr_framework(model=model, optimizer=optimizer,
                            scheduler=scheduler, args=args)
    print('simlcr loaded..')
    print('training started..')
    simclr.train(train_loader)
    print('training completed..')

if __name__ == "__main__":
    main()