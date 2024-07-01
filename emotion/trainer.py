import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import tqdm

class trainer:
    def __init__(self, model, dataset, device, val_dataset = None, metric = None, batch_size=32, num_epochs=10, lr=0.001, **kwargs):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

        self.device = device
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10)
        
        if val_dataset:
            assert metric, "Metric must be provided if validation dataset is provided"
            self.val_dataset = val_dataset
            self.metric = metric
            self.best_metric = 0.0
            self.best_model_epoch = -1
            self.val = True
        else:
            self.val = False

    def train(self):
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        print("Starting Training")
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            for i, data in enumerate(train_loader, 0):
                # print(f"Batch {i+1}/{len(train_loader)}")
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
            self.scheduler.step()
            
            if self.val:
                metric = self.metric(self.model, self.val_dataset, self.device, self.batch_size)
                print(f"Validation Metric: {metric}")
                if metric > self.best_metric:
                    self.best_metric = metric
                    self.best_model_epoch = epoch
                    torch.save(self.model.state_dict(), "best_model.pth")

        print("Finished Training")
        return self.model
