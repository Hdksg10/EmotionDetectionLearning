import argparse
import os
import random

import numpy as np
import torch
from torchvision import datasets, transforms
from emotion.data import EmotionDataset
from emotion.models.base import BaseModel
from emotion.trainer import trainer
from emotion.utils import eval_model

from emotion.models.convnext import get_model

seed = 2001
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.RandomRotation(degrees=20),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_se', action='store_true', default=False)
    parser.add_argument('--use_sim', action='store_true', default=False)
    parser.add_argument('--use_stn', action='store_true', default=False)
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()
    
    train_dataset = EmotionDataset(root_dir='./dataset/train', transform=train_transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(0.8*len(train_dataset)), len(train_dataset)-int(0.8*len(train_dataset))])
    test_dataset = EmotionDataset(root_dir='./dataset/test', transform=test_transform)
    # train_dataset = EmotionDataset(root_dir='./dataset/train', transform=None)
    # test_dataset = EmotionDataset(root_dir='./dataset/test', transform=None)
    # note input figure size is 48x48=2304
    # model = BaseModel(input_dim = 2304, num_classes=7)
    model = get_model("convnext_xlarge", num_classes=7, pretrained=True, in_22k=True, use_se=args.use_se, use_sim=args.use_sim, use_stn=args.use_stn)
    if args.ckpt:
        print(f"Loading checkpoint from {args.ckpt}")
        model.load_state_dict(torch.load(args.ckpt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = trainer(model, train_dataset, device, test_dataset, eval_model, args.batch_size, args.num_epochs, args.lr)
    model = trainer.train()
    print(f"Accuracy: {eval_model(model, test_dataset, device, args.batch_size)}")