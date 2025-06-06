import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 3, 224, 224)  # 100张假图片
        self.labels = torch.randint(0, 2, (100,))   # 2个类别

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,  stride=1, padding=0)
        self.fc = nn.Linear(in_features=222 * 222, out_features=2, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # model.to(device)

    for epoch in range(2):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")


if __name__ == "__main__":
    train()