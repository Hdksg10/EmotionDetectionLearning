import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


emotion_to_id = {
    "angry" : 0,
    "disgusted" : 1,
    "fearful" : 2,
    "happy" : 3,
    "neutral" : 4,
    "sad" : 5,
    "surprised" : 6
}

class EmotionDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        
        # Traverse the root directory to get all image paths and labels
        for _, label in enumerate(os.listdir(root_dir)):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                idx = emotion_to_id[label]
                image_names = os.listdir(label_dir)
                for image_name in tqdm(image_names, desc=f"Loading subdataset {label}"):
                    image_path = os.path.join(label_dir, image_name)
                    if os.path.isfile(image_path):
                        image = Image.open(image_path).convert('L')  # Convert to grayscale
                        # image_tensor = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255.0  # Convert to tensor
                        image_tensor = transforms.ToTensor()(image)
                        self.images.append(image_tensor)
                        self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        label = self.labels[idx]

        return image, label

if __name__ == '__main__':
    dataset = EmotionDataset(root_dir='../dataset')