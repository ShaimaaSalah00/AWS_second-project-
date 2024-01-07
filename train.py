import argparse
import torch
from torch import nn, optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset  
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import json
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = Path(root_dir)
        self.labels = self.load_labels(labels_file)
        self.filenames = [f.name for f in self.root_dir.iterdir() if f.is_dir()]  
        self.transform = transform

    def __len__(self):
        return sum((len(list((self.root_dir / folder).iterdir())) for folder in self.filenames))

    def __getitem__(self, idx):
        folder_idx = 0
        while idx >= len(list((self.root_dir / self.filenames[folder_idx]).iterdir())):
            idx -= len(list((self.root_dir / self.filenames[folder_idx]).iterdir()))
            folder_idx += 1

        folder_name = self.filenames[folder_idx]
        img_name = list((self.root_dir / folder_name).iterdir())[idx].name
        img_path = self.root_dir / folder_name / img_name
        image = Image.open(str(img_path)).convert("RGB")

        label = self.labels[folder_name]

        if self.transform:
            image = self.transform(image)

        return image, label

    def load_labels(self, labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        return labels


def train_model(data_dir, labels_file, architecture, lr, hidden_units, epochs, use_gpu):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(35),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(root=data_dir + '/' + x, transform=data_transforms[x])
                      for x in ['train', 'valid', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4)
                   for x in ['train', 'valid', 'test']}

    if architecture == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif architecture == 'resnet':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError("Invalid architecture. Choose either 'alexnet' or 'resnet'.")

    classifier = nn.Sequential(
        nn.Linear(9216, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'Epoch: {epoch + 1}/{epochs}... {phase.capitalize()} Loss: {epoch_loss:.3f}')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
        'epochs': epochs,
        'arch': architecture,
        'classifier': model.classifier  
    }, 'trained_model.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset.")
    parser.add_argument("data_dir", help="Path to the dataset directory.")
    parser.add_argument("labels_file", help="Path to the labels JSON file.")
    parser.add_argument("--architecture", choices=["alexnet", "resnet"], default="alexnet", help="Model architecture.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--hidden_units", type=int, default=4096, help="Number of hidden units.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--use_gpu", action="store_true", help="Train the model on GPU.")

    args = parser.parse_args()
    train_model(args.data_dir, args.labels_file, args.architecture, args.lr, args.hidden_units, args.epochs, args.use_gpu)
