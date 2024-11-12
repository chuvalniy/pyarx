from torch.utils.data import Dataset
import torch
import math
from PIL import Image
import os

class YOLOv1Dataset(Dataset):
    def __init__(
            self, 
            images_path: str, 
            labels_path: str,
            cell_size: int = 7,
            n_classes: int = 20,
            input_size: tuple[int, int] = (448, 448),
            transform=None,
        ):
        super().__init__()
        self.images_path = images_path
        self.labels_path = labels_path

        self.data_files = [f.replace(".txt", "") for f in os.listdir(labels_path)]

        self.cell_size = cell_size
        self.n_classes = n_classes

        self.height = input_size[0]
        self.width = input_size[1]

        self.height_step = self.height // cell_size
        self.width_step = self.width // cell_size

        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        sample = self.data_files[idx]
        image_path = os.path.join(self.images_path, sample, ".jpg")
        label_path = os.path.join(self.labels_path, sample, ".txt")

        image = Image(image_path)

        if self.transform:
            image = self.transform(image)

        label = torch.zeros(self.cell_size, self.cell_size, self.n_classes + 5)
        with open(label_path, 'r') as f:
            data = f.readlines()

            for line in data:
                cx, cy, w, h, cls = line.strip().split()

                cx_cell = (self.width * cx) / self.width_step
                cy_cell = (self.height * cy) / self.height_step

                cx_pos, cx_rel = math.modf(cx_cell)
                cy_pos, cy_rel = math.modf(cy_cell)

                label[cx_pos, cy_pos, int(cls)] = 1.0
                label[cx_pos, cy_pos, 21:25] = torch.tensor([cx_rel, cy_rel, w, h])
                label[cx_pos, cy_pos, 20] = 1.0

        return image, label