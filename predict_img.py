import torch
import torchvision.transforms as transforms
from PIL import Image

from model import Ai

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

model = Ai()

model.load_state_dict(torch.load('digits_model.pth'))
model.eval()

img = Image.open('img8.png')

tenzor_img = transform(img)
tenzor_img = tenzor_img.unsqueeze(0)

with torch.no_grad():
    outputs = model(tenzor_img)

    _, predicted = torch.max(outputs, 1)
    print("\nИИ думает, что на картинке цифра:", predicted.item())