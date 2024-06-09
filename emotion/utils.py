from torch.utils.data import DataLoader
import torch
import numpy

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