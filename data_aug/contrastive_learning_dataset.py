from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from data_aug.image_dataset import ImageDataset


class ContrastiveLearningDataset:
    def __init__(self, root_folder, force_download=False):
        self.root_folder = root_folder
        self.force_download = force_download

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {
            'unlabeled': lambda: ImageDataset(self.root_folder, force_download=self.force_download, unlabeled=True,
                                              transform=ContrastiveLearningViewGenerator(
                                                  self.get_simclr_pipeline_transform(64),
                                                  n_views)),
            'train': lambda: ImageDataset(self.root_folder, force_download=self.force_download, train=True,
                                          transform=ContrastiveLearningViewGenerator(
                                              self.get_simclr_pipeline_transform(64),
                                              n_views)),
            'validate': lambda: ImageDataset(self.root_folder, force_download=self.force_download, valid=True,
                                             transform=ContrastiveLearningViewGenerator(
                                                 self.get_simclr_pipeline_transform(64),
                                                 n_views))
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
