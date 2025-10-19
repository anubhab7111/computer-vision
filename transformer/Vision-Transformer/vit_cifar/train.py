import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import pathlib
import argparse
from model import *

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k * (100.0 / batch_size)).item())
        return res


def train_one_epoch(model, dataloader, optimizer, device, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = F.cross_entropy(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        if (i + 1) % 100 == 0:
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            print(f"Epoch {epoch} Iter {i+1}/{len(dataloader)} loss {loss.item():.4f} acc@1 {acc1:.2f}")

    avg_loss = total_loss / total_examples
    return avg_loss


def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    total_loss = 0.0
    top1 = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, targets)
            total += images.size(0)
            total_loss += loss.item() * images.size(0)
            top1 += accuracy(outputs, targets, topk=(1,))[0] * images.size(0) / 100.0

    avg_loss = total_loss / total
    acc1 = (top1 / total) * 100.0
    return avg_loss, acc1


# ---------------------
# Main: dataset and run
# ---------------------
def main():
    parser = argparse.ArgumentParser(description="ViT on CIFAR-10 example")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("Using device:", device)

    # Data transforms for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = VisionTransformer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optional simple linear LR warmup+cosine schedule
    total_steps = len(trainloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_acc = 0.0
    save_dir = pathlib.Path("checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, trainloader, optimizer, device, epoch, scheduler=None)
        val_loss, val_acc = evaluate(model, testloader, device)
        print(f"Epoch {epoch} TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} ValAcc={val_acc:.2f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
            }, save_dir / f"vit_cifar_best.pth")
            print(f"Saved best model with val_acc={val_acc:.2f}")

    print("Training finished. Best val acc: {:.2f}".format(best_acc))


if __name__ == "__main__":
    main()

