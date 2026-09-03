import torch
import subprocess
import os

from model import Ai
from dataset import test_loader

def main():
    if not os.path.exists('./data'):
        print('\nНет датасета')
        subprocess.run(['python', 'dataset.py'])
    else:
        print('\nДатасет есть')

    if not os.path.exists('digits_model.pth'):
        print('\nНет обученной модели')
        subprocess.run(['python', 'train.py'])
    else:
        print('\nОбученная модель есть')

    model = Ai()
    model.load_state_dict(torch.load('digits_model.pth'))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\nТочность модели на тестовых данных: {accuracy:.2f}%')

    print('\nТест кастомной картинки')
    subprocess.run(['python', 'predict_img.py'])

if __name__ == '__main__':
    main()