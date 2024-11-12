import torch
from src.yolo_v1 import YOLOv1, YOLOv1Loss, YOLOv1Postprocessor
from src.yolo_v2 import YOLOv2, YOLOv2Loss

if __name__ == '__main__':
    # model = YOLOv1(3)
    # criterion = YOLOv1Loss()
    # postprocessor = YOLOv1Postprocessor(threshold=1e-4)

    # x = torch.rand(4, 3, 448, 448)
    # y = torch.rand(4, 7, 7, 25)

    # pred = model(x)
    # loss = criterion(y, pred)

    # output = postprocessor(pred)
    # print(output.shape)

    model = YOLOv2(3, 20)
    criterion = YOLOv2Loss()


    x = torch.rand(4, 3, 416, 416)
    y = torch.rand(4, 13, 13, 25)

    pred = model(x)
    loss = criterion(y, pred)

