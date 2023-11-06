import torch
from ultralytics import YOLO

if __name__ == '__main__':
    device = torch.device('cuda')
    model = YOLO('yolov8n.pt').to(device)
    results = model.train(data="data.yaml", epochs=100, imgsz=640, device=device, workers=16)
    

    
