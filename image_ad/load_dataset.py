import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, split='train', transform=None):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = [] # 0 for good, 1 for anomaly

        if split == 'train':
            img_dir = os.path.join(root_dir, category, 'train', 'good')
            if os.path.exists(img_dir):
                for img_name in os.listdir(img_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        self.image_paths.append(os.path.join(img_dir, img_name))
                        self.labels.append(0)
            else:
                print(f"Warning: Directory {img_dir} does not exist.")
        else: # test
            test_dir = os.path.join(root_dir, category, 'test')
            if os.path.exists(test_dir):
                for defect_type in os.listdir(test_dir):
                    defect_dir = os.path.join(test_dir, defect_type)
                    if os.path.isdir(defect_dir):
                        label = 0 if defect_type == 'good' else 1
                        for img_name in os.listdir(defect_dir):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                                self.image_paths.append(os.path.join(defect_dir, img_name))
                                self.labels.append(label)
            else:
                print(f"Warning: Directory {test_dir} does not exist.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_mvtec_loader(root_dir, category, batch_size=32, image_size=256, split='train'):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = MVTecDataset(root_dir, category, split, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
    return dataloader
