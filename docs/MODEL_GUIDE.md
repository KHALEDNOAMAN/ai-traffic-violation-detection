# AI Traffic Violation Detection - Model Guide

## Detection Pipeline
```
Video Feed --> Frame Extraction --> YOLO Detection --> Violation Classification --> Alert
                                       |
                              Vehicle + License Plate
                                       |
                              Speed Estimation + Lane Check
```

## Violation Types
| Violation | Detection Method | Accuracy |
|-----------|-----------------|----------|
| Red light running | Traffic light + vehicle position | ~94% |
| Speeding | Frame-to-frame distance calculation | ~89% |
| Wrong lane | Lane boundary detection | ~91% |
| No helmet | Rider classification | ~93% |

## Model Architecture
- Base: YOLOv8 (object detection)
- Custom classes trained on traffic dataset
- OCR for license plate reading

## Training Your Own Model
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="traffic.yaml", epochs=100, imgsz=640)
```

## Performance
| Metric | Value |
|--------|-------|
| mAP@0.5 | 92.3% |
| FPS (GPU) | ~25 |
| FPS (CPU) | ~4 |