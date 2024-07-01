from torch.utils.data import DataLoader
import torch
import numpy as np

from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from captum.attr import Occlusion
from captum.attr import visualization as viz

emotion_to_id = {
    "angry" : 0,
    "disgusted" : 1,
    "fearful" : 2,
    "happy" : 3,
    "neutral" : 4,
    "sad" : 5,
    "surprised" : 6
}


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
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(224),
            # transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

def grad_cam(model, image, category):
    image_tensor = test_transform(image)
    input_tensor = torch.unsqueeze(image_tensor, dim=0)

    image = transforms.ToPILImage()(image_tensor)
    model.eval()
    # target_layers 选择最后一个卷积层
    # target_layers = [model.conv4[-1]]
    # target_layers = [model.features[-1]]
    # target_layers = [model.layer4[-1]]
    target_layers = [model.downsample_layers[3][1]]
    # target_layers = [model.head]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, aug_smooth=True)
    grayscale_cam = grayscale_cam[0, :]

    image = np.array(image.convert('RGB'), dtype=np.float32)
    print(image.shape)
    visualization = show_cam_on_image(image / 255., grayscale_cam, use_rgb=True)
    plt.title(category)
    plt.imshow(visualization)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.savefig("grad_cam.png")
    
def occulusion(model, image, category):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_resized = test_transform(image)
    image_input = torch.unsqueeze(image_resized, dim=0).to(device)
    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(image_input,
                                       strides = (3, 8, 8),
                                       target=emotion_to_id[category],
                                       sliding_window_shapes=(3,28, 28),
                                       baselines=0)
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      np.transpose(image_resized.squeeze().cpu().detach().numpy(), (1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      show_colorbar=True,
                                      outlier_perc=2)
    plt.savefig('attributions.png')