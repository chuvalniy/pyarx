from torch.utils.data import Dataset
import torch
import math
from PIL import Image
import os


class YOLOv3Dataset(Dataset):
    def __init__(
            self, 
            images_path: str, 
            labels_path: str,
            anchors: tuple[tuple[float]],
            cell_sizes: tuple[int] = (13, 26, 52),
            n_classes: int = 80,
            image_size: tuple[int, int] = (448, 448),
            transform=None,
        ):
        super().__init__()
        self.images_path = images_path
        self.labels_path = labels_path
        self.data_files = [f.replace(".txt", "") for f in os.listdir(labels_path)]

        self.n_classes = n_classes
        self.height = image_size[0]
        self.width = image_size[1]
        self.cell_sizes = cell_sizes
        self.anchors = anchors

        self.transform = transform

    def __getitem__(self, idx):
        sample = self.data_files[idx]
        image_path = os.path.join(self.images_path, sample, ".jpg")
        label_path = os.path.join(self.labels_path, sample, ".txt")
    # Take file by idx
    # Init output tensor (3, n_boxes, cell_size, cell_size, 6)
    # Open label file, parse (cls, cx, cy, w, h)
    # For each scale find best box between anchor and y_true (by IoU)
    # Find coordinates for a box (x_cell, y_cell)

    def __len__(self):
        return len(self.data_files)
