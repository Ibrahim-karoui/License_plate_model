import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(BASE_DIR, "runs/detect/train/results.csv")

results = pd.read_csv(results_path, sep=";")
results.columns = results.columns.str.strip()


plt.figure()
plt.plot(results['epoch'], results['train/box_loss'], label='train/box_loss')
plt.plot(results['epoch'], results['val/box_loss'], label='val/box_loss', c='red')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('box_loss')
plt.title('box_loss vs epochs')
plt.legend()

plt.figure()
plt.plot(results['epoch'], results['train/cls_loss'], label='train/cls_loss')
plt.plot(results['epoch'], results['val/cls_loss'], label='val/cls_loss', c='red')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('cls_loss')
plt.title('cls_loss vs epochs')
plt.legend()

plt.figure()
plt.plot(results['epoch'], results['train/dfl_loss'], label='train/dfl_loss')
plt.plot(results['epoch'], results['val/dfl_loss'], label='val/dfl_loss', c='red')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('dfl_loss')
plt.title('dfl_loss vs epochs')
plt.legend()

plt.figure()
plt.plot(results['epoch'], results['train/dfl_loss'] + results['train/cls_loss'] + results['train/box_loss'] , label='train/total_loss')
plt.plot(results['epoch'], results['val/dfl_loss'] + results['val/cls_loss'] + results['val/box_loss'] , label='val/total_loss', c='red')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('total_loss')
plt.title('total_loss vs epochs')
plt.legend()


plt.figure()
plt.plot(results['epoch'], results['metrics/precision(B)'], label='Precision')
plt.plot(results['epoch'], results['metrics/recall(B)'], label='Recall')
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.title("Precision & Recall vs Epochs")
plt.grid()
plt.legend()

plt.figure()
plt.plot(results['epoch'], results['metrics/mAP50(B)'], label='metrics/mAP50(B)')
plt.plot(results['epoch'], results['metrics/mAP50-95(B)'], label='metrics/mAP50-95(B)')
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.title("mAP50(B) & mAP50-95(B) vs Epochs")
plt.grid()
plt.legend()


plt.show()
