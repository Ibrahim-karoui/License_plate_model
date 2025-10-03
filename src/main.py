import os
from ultralytics import YOLO

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #print(BASE_DIR)
    config_path = os.path.join(BASE_DIR, "config.yaml")
    #print(config_path)
    
    model = YOLO("yolov8m.yaml")
    results = model.train(data = config_path, epochs = 30, resume = True, device = 0)
    metrics = model.val(data = config_path, split = "test")
    print(metrics)

if __name__ == "__main__" :
    main()