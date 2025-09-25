# License Plate Detection Model

This repository contains a YOLOv8-based license plate detection model trained on a custom dataset. The project includes training scripts, configuration files, and trained model weights.

## Project Structure

- `src/` : Python scripts for training and plotting metrics  
- `data/` : Dataset folders (`train`, `valid`, `test`) with images and labels  
- `runs/` : Training outputs including weights (`best.pt`, `last.pt`) and metrics  
- `config.yaml` : Dataset configuration for YOLOv8  
- `requirements.txt` : Python dependencies  

## Dataset

The dataset used for training and validation is available [here](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4/download).

**Note:** The dataset is not included in this repository due to size constraints.
