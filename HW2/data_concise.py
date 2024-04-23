import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def load_data_concise(root="cifar10", batch_size=64):
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)
    train_trans = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5), T.Normalize(MEAN, STD)])
    test_trans = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    train_data = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_trans)
    test_data = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_trans)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = load_data_concise(root="cifar10", batch_size=64)
    for i, (images, labels) in enumerate(test_loader):
        print(images[0])
        break
