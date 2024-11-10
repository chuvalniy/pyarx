import torch

from src.yolo_v1 import YOLOv1, YOLOv1Loss, YOLOv1Postprocessor

if __name__ == '__main__':
    model = YOLOv1(3)
    criterion = YOLOv1Loss()
    postprocessor = YOLOv1Postprocessor(threshold=1e-4)

    x = torch.rand(4, 3, 448, 448)
    y = torch.rand(4, 7, 7, 25)

    pred = model(x)
    loss = criterion(y, pred)

    output = postprocessor(pred)
    print(output.shape)
