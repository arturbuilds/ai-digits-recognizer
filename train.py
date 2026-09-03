import torch.nn as nn
import torch
import numpy as np

from model import Ai
from dataset import train_loader

model = Ai()   

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()

epochs = 5

for epoch in range(1, epochs + 1):
    for images, labels in train_loader:
        pred = model(images)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Эпоха {epoch}/{epochs} | Ошибка (Loss): {loss.item():.4f}")

print('\nОбучение завершено!')

torch.save(model.state_dict(), 'digits_model.pth')
print('\nМодель сохраненна!')