from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt") # Il le télécharge s'il ne le trouve pas

results = model.train(
    data="config.yaml",
    epochs=2,
    imgsz=640,
    batch=2,
    workers=0,
    device=0,
    amp=False
) # mettre des trucs plus énervé pour le serveur 