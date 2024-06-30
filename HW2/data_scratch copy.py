import os
from math import ceil

import torch
import torchvision

torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_cifar10(root="cifar10", train=True):
    """直接将整个数据集载入到 GPU 上, 可以加快训练速度"""
    save_path = os.path.join(root, "train.pt" if train else "test.pt")
    if not os.path.exists(save_path):
        dset = torchvision.datasets.CIFAR10(root, download=True, train=train)
        images = torch.tensor(dset.data, dtype=torch.float32)
        labels = torch.tensor(dset.targets, dtype=torch.int64)
        torch.save({"images": images, "labels": labels}, save_path)
    data = torch.load(save_path, map_location=device)
    return data["images"], data["labels"]


class ToTensor:
    """将 [0, 255] 的像素值缩放到 [0.0, 1.0], 维度顺序从 (B, H, W, C) 转换为 (B, C, H, W)
    Hint: 维度转换可以使用 torch.Tensor.permute 函数, https://pytorch.org/docs/stable/generated/torch.permute.html#torch.permute

    Args:
        imgs (torch.Tensor): Byte tensor image of size (B, H, W, C) to be converted to tensor.

    Returns:
        torch.Tensor: Float tensor image of size (B, C, H, W).
    """

    def __call__(self, imgs: torch.Tensor):
        imgs = imgs.float() / 255.0
        imgs = imgs.permute(0, 3, 1, 2)
        return imgs


class RandomHorizontalFlip:
    """对输入的每一张图片, 以独立的概率 p 进行水平翻转, 注意使用批量处理的方式而不是逐个处理每张图片.
    Hint: 翻转可以使用 torch.flip 函数, https://pytorch.org/docs/stable/generated/torch.flip.html

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        imgs (torch.Tensor): Float tensor image of size (B, C, H, W) to be flipped.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs: torch.Tensor):
        batch_size = imgs.size(0)
        imgs_copy = imgs.clone()
        # 生成一个随机数组，数组中的每个元素都是0或1, 1表示需要翻转
        flip_mask = torch.rand(batch_size, device=imgs.device) < self.p
        # 对需要翻转的图片进行水平翻转
        imgs_copy[flip_mask] = torch.flip(imgs[flip_mask], dims=[3])
        return imgs_copy


class Normalize:
    """使用 mean 和 std 对张量图像进行归一化.

    Args:
        mean (tuple): A tuple of mean values for each channel.
        std (tuple): A tuple of standard deviation values for each channel.
        imgs (torch.Tensor): Float tensor image of size (B, C, H, W) to be normalized.
    """

    def __init__(self, mean: tuple, std: tuple):
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).reshape(1, 3, 1, 1)

    def __call__(self, imgs: torch.Tensor):
        imgs = (imgs - self.mean) / self.std
        return imgs


class Compose:
    """按列表中的顺序依次对图像进行处理.

    Args:
        transforms (list): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for transform in self.transforms:
            imgs = transform(imgs)
        return imgs


class CifarDataset:
    def __init__(self, root, train=True, transform=None):
        self.images, self.labels = load_cifar10(root, train=train)
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]
        labels = self.labels[idx]
        if self.transform is not None:
            images = self.transform(images)
        return images, labels


class CifarLoader:
    def __init__(self, dataset: CifarDataset, batch_size=512, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.train = dataset.train
        if self.train and shuffle:
            self.shuffle = True
        else:
            self.shuffle = False
        self.data_num = len(dataset)

    def __len__(self):
        """返回 batch 的个数"""
        return ceil(self.data_num / self.batch_size)

    def __iter__(self):
        """使用迭代器返回 image 和 label"""
        indices = torch.randperm(self.data_num) if self.shuffle else torch.arange(self.data_num)
        for i in range(len(self)):
            idxs = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.dataset[idxs]


def load_data_scrach(root="cifar10", batch_size=64):
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)
    train_trans = Compose([ToTensor(), RandomHorizontalFlip(0.5), Normalize(MEAN, STD)])
    test_trans = Compose([ToTensor(), Normalize(MEAN, STD)])

    train_data = CifarDataset(root=root, train=True, transform=train_trans)
    test_data = CifarDataset("cifar10", train=False, transform=test_trans)

    train_loader = CifarLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = CifarLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = load_data_scrach(root="cifar10", batch_size=64)
    for i, (images, labels) in enumerate(test_loader):
        print(images[0])
        break
