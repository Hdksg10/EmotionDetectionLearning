from torch.utils.data import DataLoader
import torch
import numpy as np

from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms


def eval_model(model, dataset, device, batch_size=32):
    model.eval()
    correct = 0
    total = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def grad_cam(model, image, category):
    image_tensor = transforms.ToTensor()(image)
    input_tensor = torch.unsqueeze(image_tensor, dim=0)

    model.eval()
    # target_layers 选择最后一个卷积层
    # target_layers = [model.conv4[-1]]
    # target_layers = [model.features[-1]]
    # target_layers = [model.layer4[-1]]
    target_layers = [model.dense4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, aug_smooth=True)
    grayscale_cam = grayscale_cam[0, :]

    image = np.array(image.convert('RGB'), dtype=np.float32)
    visualization = show_cam_on_image(image / 255., grayscale_cam, use_rgb=True)
    plt.title(category)
    plt.imshow(visualization)
    plt.xticks([])
    plt.yticks([])
    plt.show()
