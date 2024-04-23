import numpy as np
import torch
import torchvision.transforms as T

from data_scratch import Compose, Normalize, RandomHorizontalFlip, ToTensor


def test_ToTensor():
    imgs = np.random.randint(0, 256, (10, 32, 32, 3), dtype=np.uint8)
    tensor_imgs = ToTensor()(torch.tensor(imgs))
    expected_imgs = torch.stack([T.ToTensor()(img) for img in imgs], dim=0)
    assert torch.allclose(tensor_imgs, expected_imgs, atol=1e-5)
    print("ToTensor test passed.")


def test_RandomHorizontalFlip():
    p = 0.35
    N = 1000
    imgs = torch.randn(N, 3, 32, 32)
    flipped_imgs = RandomHorizontalFlip(p)(imgs)

    assert flipped_imgs.shape == imgs.shape, "Output shape should be the same as input shape."
    assert isinstance(flipped_imgs, torch.Tensor), "Output should be a torch.Tensor."

    expected_flips = int(p * N)
    actual_flips = sum(not torch.equal(imgs[i], flipped_imgs[i]) for i in range(N))
    diff_flips = abs(actual_flips - expected_flips)
    assert diff_flips < np.sqrt(N), f"Expected {expected_flips} flips, got {actual_flips} flips."

    p = 1
    imgs = torch.randn(N, 3, 32, 32)
    flipped_imgs = RandomHorizontalFlip(p)(imgs)
    expected_imgs = T.RandomHorizontalFlip(p)(imgs)
    # expected_imgs = torch.flip(imgs, dims=[3])
    assert torch.allclose(flipped_imgs, expected_imgs)
    print("RandomHorizontalFlip test passed.")


def test_Normalize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs = torch.randn(10, 3, 32, 32, device=device) * 2 + 1
    mean = (0.1, 0.2, 0.3)
    std = (0.4, 0.5, 0.6)
    normalized_imgs = Normalize(mean, std)(imgs)
    expected_imgs = T.Normalize(mean, std)(imgs)
    assert torch.allclose(normalized_imgs, expected_imgs, atol=1e-5)
    print("Normalize test passed.")


def test_Compose():
    input = torch.randn(5)
    f1 = lambda x: x * 2
    f2 = lambda x: x + 1
    f = Compose([f1, f2])
    output = f(input)
    expected_output = f2(f1(input))
    assert torch.allclose(output, expected_output, atol=1e-5)
    print("Compose test passed.")


if __name__ == "__main__":
    test_Normalize()
    test_ToTensor()
    test_RandomHorizontalFlip()
