# Accelerated JKO

# Set up

```bash 
conda create -n (name your env here) python=3.11
pip install -r requirements.txt
```

#### Installing torch

- If on CPU only:
  
```bash
pip install torch torchvision
```

- else:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

This is an example, find your install command here: https://pytorch.org/get-started/locally/

#### Installing MNISTDIffusion

From project root, run:

```bash
git clone https://github.com/bot66/MNISTDiffusion.git
```