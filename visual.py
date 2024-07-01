import argparse
import os
import random
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms


from emotion.data import EmotionDataset
from emotion.models.base import BaseModel
from emotion.trainer import trainer
from emotion.utils import eval_model, grad_cam, occulusion

from emotion.models.convnext import get_model

seed = 2001
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(224),
            # transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_se', action='store_true', default=False)
    parser.add_argument('--use_sim', action='store_true', default=False)
    parser.add_argument('--use_stn', action='store_true', default=False)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--label', type=str, default=None)
    args = parser.parse_args()
    
    model = get_model("convnext_xlarge", num_classes=7, pretrained=True, in_22k=True, use_se=args.use_se, use_sim=args.use_sim, use_stn=args.use_stn)
    print(f"Loading checkpoint from {args.ckpt}")
    model.load_state_dict(torch.load(args.ckpt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image = Image.open(args.input).convert('L')
    grad_cam(model, image, args.label)
    model = model.to(device)
    
    # show occlusion output
    occulusion(model, image, args.label)
    
    # show STN output
    # image_resized = transforms.ToPILImage()(image_resized)
    # image_resized.save("resized_image.png")
    # stn_image = model.forward_stn(test_transform(image).unsqueeze(0))
    # stn_image = transforms.ToPILImage()(stn_image.squeeze(0).cpu())
    # stn_image.save("stn_image.png")
