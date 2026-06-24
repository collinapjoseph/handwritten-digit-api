import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Configuration
BATCH_SIZE = 64
EPOCHS     = 5
LR         = 1e-3
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data and 
# Augment using RandomAffine()
# Normalize using commonly used mean and std for this dataset
# Ref: Line 118 - https://github.com/pytorch/examples/blob/main/mnist/main.py
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=10, translate=(0.3, 0.3), scale=(0.5, 1.5), shear=5),
    transforms.Normalize((0.1307,), (0.3081,)), 
])

train_data = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

# Training & evaluation
criterion = nn.CrossEntropyLoss()

# Models
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),                           
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def train(model, optimizer, epoch, label):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"[{label}] Epoch {epoch}/{EPOCHS}  loss: {total_loss / len(train_loader):.4f}")

def evaluate(model, label):
    model.eval()
    num_classes = 10
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=1)
            for true, pred in zip(labels.cpu(), preds.cpu()):
                matrix[true][pred] += 1

    accuracy = matrix.diagonal().sum().item() / len(test_data) * 100
    print(f"[{label}] Test accuracy: {accuracy:.2f}%")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix.numpy(), cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{label} — Confusion Matrix")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    plt.colorbar(im, ax=ax)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, matrix[i, j].item(),
                    ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > matrix.max() * 0.5 else "black")
    plt.tight_layout()
    fname = f"confusion_matrix_{label.lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {fname}")

    return accuracy

def save_onnx(model, path):
    model.eval()
    dummy = torch.zeros(1, 1, 28, 28, device=DEVICE)
    torch.onnx.export(
        model, dummy, path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"Model saved to {path}")


# Train CNN
print("\n── Training CNN ──")
cnn = CNN().to(DEVICE)
opt3 = optim.Adam(cnn.parameters(), lr=LR)
for epoch in range(1, EPOCHS + 1):
    train(cnn, opt3, epoch, "CNN")
acc3 = evaluate(cnn, "CNN")

print(f"CNN accuracy: {acc3:.2f}%")

# Save model
save_onnx(cnn, "mnist_cnn.onnx")
