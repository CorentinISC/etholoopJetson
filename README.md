## Wrapper C++ pour YOLOv8 (pour Jetson Jetpack 4.X)

### Setup de la Jetson

### Setup du modèle

Nécessite un modèle en `.engine`.

Pour cela on va faire `.pt` -> `.onnx` -> `.engine`.

On peut télécharger les `.pt` par exemple [ici](https://docs.ultralytics.com/models/yolov8/#performance-metrics).

Pour convertir un `.pt` en `.onnx` on utilise un code python du type :

```
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.export(format="onnx")
```

Pour convertir un `.onnx` en `.engine` (tensorrt) on utilise l'éxecutable de NVIDIA :

```
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolov8s.onnx \
    --saveEngine=yolov8s.engine \
    --fp16 \
    --workspace=2048
```

Le code main présente un exemple d'utilisation du wrapper avec l'API Ximea (choix non optimal au vue du délai d'acquisition, préférer OpenCV).