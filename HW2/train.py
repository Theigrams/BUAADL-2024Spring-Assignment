import torch
import torch.nn as nn
from data_concise import load_data_concise
from data_scratch import load_data_scrach
from utils import Accumulator, SimpleCNN, print_columns, print_training_details

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True


def train_epoch(net, train_iter, loss_fn, optimizer):
    net.train()
    device = next(net.parameters()).device
    metrics = Accumulator(4)
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics.add(loss * len(y), accuracy(y_hat, y) * len(y), len(y))
        train_loss = metrics[0] / metrics[2]
        train_acc = metrics[1] / metrics[2]
    return train_loss, train_acc


@torch.no_grad()
def eval_model(net, test_iter, loss_fn):
    net.eval()
    device = next(net.parameters()).device
    metrics = Accumulator(3)
    for X, y in test_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        metrics.add(loss * len(y), accuracy(y_hat, y) * len(y), len(y))
    test_loss = metrics[0] / metrics[2]
    test_acc = metrics[1] / metrics[2]
    return test_loss, test_acc


def accuracy(y_hat, y_true):
    y_pred = y_hat.argmax(dim=1)
    return (y_pred == y_true).float().mean().item()


def run_train(config):
    epochs = config["train_epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    use_scratch = config["use_scratch"]
    if use_scratch:
        train_loader, test_loader = load_data_scrach("cifar10", batch_size)
    else:
        train_loader, test_loader = load_data_concise("cifar10", batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # 用于准确地测量GPU上的操作时间
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    print_columns(is_head=True)
    for epoch in range(1, epochs + 1):
        starter.record()
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer)
        ender.record()
        torch.cuda.synchronize()  # 等待所有CUDA操作完成
        total_time += 1e-3 * starter.elapsed_time(ender)

        test_loss, test_acc = eval_model(model, test_loader, loss_fn)
        # 打印训练和测试的损失和准确性
        print_training_details(locals(), is_final_entry=True if epoch == epochs else False)


if __name__ == "__main__":
    config = {
        "train_epochs": 10,
        "batch_size": 512,
        "lr": 1e-3,
        "use_scratch": True,
    }
    print("Using scratch data loader")
    run_train(config)

    print("Using concise data loader")
    config["use_scratch"] = False
    run_train(config)
