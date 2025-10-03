import torch

print(torch.__version__)         # affiche la version de PyTorch
print(torch.version.cuda)        # affiche la version de CUDA supportée
print(torch.cuda.is_available()) # True si ton GPU est détecté
