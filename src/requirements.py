import subprocess
import sys

# Install torchvision from the PyTorch index with force-reinstall and no-cache-dir
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "torchvision",
                       "--force-reinstall",
                       "--no-cache-dir",
                       "--index-url", "https://download.pytorch.org/whl/cu118"])

# Install ultralytics from PyPI
subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])

# Install pandas
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
