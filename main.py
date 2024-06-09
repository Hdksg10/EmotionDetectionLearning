import argparse
import os

import torch
from emotion.data import EmotionDataset
from emotion.models.base import BaseModel
from emotion.trainer import trainer
from emotion.utils import eval_model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    train_dataset = EmotionDataset(root_dir='./dataset/train')
    test_dataset = EmotionDataset(root_dir='./dataset/test')
    # note input figure size is 48x48=2304
    model = BaseModel(input_dim = 2304, num_classes=7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = trainer(model, train_dataset, device, args.batch_size, args.num_epochs, args.lr)
    model = trainer.train()
    print(f"Accuracy: {eval_model(model, test_dataset, device, args.batch_size)}")