from torchvision.transforms import transforms
from gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from get_dataset_util import ViewGen


class GetTransformedDataset:
    @staticmethod
    def get_simclr_transform(size, s=1):
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply(
                                                  [color_jitter], p=0.8),
                                              transforms.RandomGrayscale(
                                                  p=0.2),
                                              GaussianBlur(
                                                  kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_cifar10_train(self, n_views):
        return datasets.CIFAR10('.', train=True,
                                transform=ViewGen(
                                    self.get_simclr_transform(
                                        32),
                                    n_views),
                                download=True)
    
    def get_cifar10_test(self, n_views):
        return datasets.CIFAR10('.', train=False,
                                transform=transforms.ToTensor(),
                                download=True)

    def get_cifar100_test(self, n_views):
        return datasets.CIFAR100('.', train=False,
                                 transform=transforms.ToTensor(),
                                 download=True)
