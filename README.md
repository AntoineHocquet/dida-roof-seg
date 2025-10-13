# dida-roof-seg
Roof segmentation (DL exercise by Dida)

This task was proposed by Dida as a recruitment exercise. The goal is to segment roofs from aerial images using deep learning techniques.

## Dataset

The dataset consists of 30 aerial images of size 512x512 pixels, along with 25 corresponding masks indicating the roof areas. The images are in JPG format, and the masks are in PNG format.
The data can be found at https://dida.do/downloads/dida-test-task.

## Instructions from Dida

There are 30 satellite pictures of houses and 25 corresponding labels that indicate the roofs. Take those 25 data points and train a neural network on them - you are completely free about the architecture and are of course allowed to use any predefined version of networks, however, you should be able to explain what you are doing - in terms of code as well as in terms of why certain steps are good choices. The preferred language is Python, but you can also use other languages. Please evaluate your network on the 5 remaining test images by making predictions of the roofs - send us the predictions and ideally some comments on what you have been doing. Everything else we will discuss from there. 

## Structure of the repository

dida-roof-seg/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  └─ raw/                     # unzip the 30 images + 25 masks here
├─ models/
│  └─ checkpoints/             # best.pth saved here
├─ outputs/
│  └─ predictions/             # 5 PNG masks for submission
└─ src/
   ├─ dida_roofseg/
   │  ├─ __init__.py
   │  ├─ seed.py               # set all RNG seeds, torch determinism
   │  ├─ io.py                 # listing files, reading images/masks, saving masks
   │  ├─ dataset.py            # RoofDataset class (train/val/test modes)
   │  ├─ model.py              # Encoder+Decoder+SegmentationModel
   │  └─ engine.py             # Trainer and Predictor
   ├─ train.py                 # CLI: trains, saves best.pth, prints metrics
   └─ predict.py               # CLI: loads best.pth, writes 5 test masks

## Object-Oriented style

The pipeline follows a lightweight OOP design (Dataset, Model, Trainer, Predictor) to encapsulate data loading, training, and inference logic. This makes it easy to extend later (e.g., with augmentation, k-fold CV, or experiment tracking) while keeping the current code minimal and readable.