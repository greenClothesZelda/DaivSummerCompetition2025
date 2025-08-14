import torch
import yaml
from types import SimpleNamespace
from torch.utils.data import ConcatDataset

from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR


def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = SimpleNamespace(**config)

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed_all(config.seed)
    else:
        device = 'cpu'

    torch.manual_seed(config.seed)

    dataset = ContrastiveLearningDataset(config.data['root_folder'], config.data['force_download'])

    unlabeled_dataset = dataset.get_dataset('unlabeled', config.data['n_views'])
    train_dataset = dataset.get_dataset('train', config.data['n_views'])
    train_dataset = ConcatDataset([unlabeled_dataset, train_dataset])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=config.arch, out_dim=config.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader),
                                                         eta_min=0, last_epoch=-1)

    # For logging purposes
    config.device = device
    config.n_views = config.data['n_views']

    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=config)
    simclr.train(train_loader)


if __name__ == "__main__":
    main()
