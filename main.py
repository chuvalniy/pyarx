import torch

from src.yolo_v1 import YOLOv1, YOLOv1Loss

if __name__ == '__main__':
    model = YOLOv1(3)
    criterion = YOLOv1Loss()

    x = torch.rand(4, 3, 448, 448)
    y = torch.rand(4, 7, 7, 25)

    pred = model(x)
    loss = criterion(y, pred)

    print(loss)
