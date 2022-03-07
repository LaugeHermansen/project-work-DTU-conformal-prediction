import torch
import numpy as np
from torch import nn
# from tqdm import tqdm, trange
from lauges_tqdm import tqdm


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

class CNN_class(nn.Module):
    def __init__(self, width, depth, input_features=28, n_classes=62):
        super().__init__()

        self.input_features = input_features
        self.width = int(width)
        self.depth = int(depth)

        layers = []
        layers.append(nn.Conv2d(1, self.width * 2, kernel_size=3, padding=1))
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.ReLU(inplace=True))

        for i in range(2, self.depth+1):
            layers.append(nn.Conv2d(self.width*2**(i-1), self.width*2**i, kernel_size=3, padding=1))
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.ReLU(inplace=True))

        temp = self.input_features
        for i in range(self.depth):
            temp = temp // 2

        self.CNN = nn.Sequential(*layers)
        #print(width*2**self.depth*temp**2)
        self.linear = nn.Linear(self.width*2**self.depth*temp**2, n_classes)

    def forward(self, x):
        x = self.CNN(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


def train(model, dataloader, lr, weight_decay, n_epochs=10):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in tqdm(range(n_epochs)):
        for im, label in tqdm(dataloader):
            im = torch.permute(im, (0, 1, 3, 2))
            optimizer.zero_grad()
            im = im.to(DEVICE)
            label = label.to(DEVICE)


            preds = model(im)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()


def test(model, dataloader):
    corrects = []
    model.eval()
    with torch.no_grad():
        for im, label in dataloader:
            im = torch.permute(im, (0, 1, 3, 2))
            im = im.to(DEVICE)
            label = label.to(DEVICE)
            corrects.append(model(im).argmax(dim=1) == label)

        acc = torch.cat(corrects).detach().cpu().numpy().mean()
    return acc
