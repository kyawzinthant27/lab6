import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

DATA = Path("realwaste-main/RealWaste")
OUTPUT = Path("results")

SEED = 42
EPOCHS = 10
BATCH_SIZE = 32
WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def make_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    blur_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.GaussianBlur(kernel_size=5, sigma=1.0),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_data = datasets.ImageFolder(DATA, train_transform)
    test_data = datasets.ImageFolder(DATA, test_transform)
    blur_data = datasets.ImageFolder(DATA, blur_transform)

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(
        len(train_data), generator=generator
    ).tolist()

    split = int(0.7 * len(indices))
    train_indices = indices[:split]
    test_indices = indices[split:]

    train_loader = DataLoader(
        Subset(train_data, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        generator=torch.Generator().manual_seed(SEED),
    )

    test_loader = DataLoader(
        Subset(test_data, test_indices),
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
    )

    blur_loader = DataLoader(
        Subset(blur_data, test_indices),
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
    )

    return train_loader, test_loader, blur_loader, train_data.classes


def make_cnn(batch_norm=False):
    layers = []

    for inputs, outputs in [(3, 16), (16, 32), (32, 64)]:
        layers.append(
            nn.Conv2d(inputs, outputs, 3, stride=2, padding=1)
        )

        if batch_norm:
            layers.append(nn.BatchNorm2d(outputs))

        layers.append(nn.ReLU())

    layers.append(nn.Conv2d(64, 9, kernel_size=25))
    layers.append(nn.Flatten())

    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    count = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        probabilities = logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        count += labels.size(0)

    return 100 * correct / count


def train_model(model, learning_rate):
    torch.manual_seed(SEED)

    train_loader, test_loader, blur_loader, _ = make_loaders()
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )
    loss_function = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        count = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = loss_function(logits, labels)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * labels.size(0)
            count += labels.size(0)

        losses.append(loss_sum / count)
        accuracy = evaluate(model, test_loader)

        print(
            f"Epoch {epoch:02d}: train loss={losses[-1]:.4f}, "
            f"test accuracy={accuracy:.2f}%",
            flush=True,
        )

    blur_accuracy = evaluate(model, blur_loader)

    print(f"Final test accuracy: {accuracy:.2f}%", flush=True)

    return losses, accuracy, blur_accuracy


def main():
    OUTPUT.mkdir(exist_ok=True)

    torch.set_num_threads(
        int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train, test, _, classes = make_loaders()

    print("Device:", DEVICE, flush=True)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    print("Classes:", classes)
    print("Training images:", len(train.dataset))
    print("Test images:", len(test.dataset))
    print(f"Seed={SEED}, epochs={EPOCHS}, batch size={BATCH_SIZE}")
    print("Image size: 200x200")
    print("Questions 1-2: Adam lr=0.001")
    print("Question 4: Adam lr=0.0001", flush=True)

    results = {}

    for question, name in [(1, "baseline"), (2, "modified")]:
        print(f"\n[Question {question}] {name} CNN", flush=True)

        torch.manual_seed(SEED)
        model = make_cnn(batch_norm=(name == "modified"))

        results[name] = train_model(model, learning_rate=0.001)
        del model

    print("\n[Question 3] Compare baseline and modified CNN")

    for name, style in [("baseline", "-"), ("modified", "--")]:
        losses, clean, blurred = results[name]

        print(f"{name}: final training loss={losses[-1]:.4f}")
        print(
            f"Clean={clean:.2f}%, blurred={blurred:.2f}%, "
            f"drop={clean - blurred:.2f} percentage points",
            flush=True,
        )

        plt.plot(
            range(1, EPOCHS + 1),
            losses,
            style,
            color="black",
            label=name,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "q3_comparison.png")
    plt.close()

    print("Loss plot: results/q3_comparison.png", flush=True)

    print("\n[Question 4] Fine-tune pretrained ResNet-18", flush=True)

    torch.manual_seed(SEED)

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )
    model.fc = nn.Linear(model.fc.in_features, 9)

    print("ImageNet weights; all layers are trainable.", flush=True)

    train_model(model, learning_rate=0.0001)

    print("\nFinished.", flush=True)


if __name__ == "__main__":
    main()
