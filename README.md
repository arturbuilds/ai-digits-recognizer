# Custom Ai Digits Recognizer

A production-ready, pure handwritten digit recognition system built with **PyTorch** and **NumPy**. The model achieves high accuracy on the MNIST dataset and supports custom image inference.

## Features
- **Pure ML Implementation:** Built from scratch without high-level wrappers.
- **Custom Image Testing:** Supports custom handwritten images drawn in Paint/Photoshop.
- **High Performance:** Reached **98.61% Test Accuracy** using the Adam optimizer.

## Tech Stack
- **Framework:** PyTorch (Core, torchvision)
- **Data Manipulation:** NumPy
- **Image Processing:** Pillow (PIL)

## Architecture
- `nn.Flatten()` (Transforms 28x28 images into a 784-dimensional vector)
- `nn.Linear(784, 512)` + `nn.ReLU()` (Upgraded hidden layer for better feature extraction)
- `nn.Linear(512, 10)` (Outputs logit probabilities for 10 classes)

## Results
- **Training Epochs:** 5
- **Final Test Accuracy:** **98.61%**
