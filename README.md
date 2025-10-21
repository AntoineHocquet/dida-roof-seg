# dida-roof-seg
Roof segmentation (DL exercise by Dida)

This task was proposed by Dida as a recruitment exercise. The goal is to segment roofs from aerial images using deep learning techniques.

## Dataset

The dataset consists of 30 aerial images of size 256x256 pixels, along with 25 corresponding masks indicating the roof areas. The images are in JPG format, and the masks are in PNG format.
The data can be found at https://dida.do/downloads/dida-test-task.

## Instructions from Dida

There are 30 satellite pictures of houses and 25 corresponding labels that indicate the roofs. Take those 25 data points and train a neural network on them - you are completely free about the architecture and are of course allowed to use any predefined version of networks, however, you should be able to explain what you are doing - in terms of code as well as in terms of why certain steps are good choices. The preferred language is Python, but you can also use other languages. Please evaluate your network on the 5 remaining test images by making predictions of the roofs - send us the predictions and ideally some comments on what you have been doing. Everything else we will discuss from there. 


## Object-Oriented style

The pipeline follows a lightweight OOP design (Dataset, Model, Trainer, Predictor) to encapsulate data loading, training, and inference logic. This makes it easy to extend later (e.g., with augmentation, k-fold CV, or experiment tracking) while keeping the current code minimal and readable.

## Project Structure

```dida-roof-seg
.
├── data
│  └── raw
|     ├── images        # input images (PNGs 256x256)
|     └── masks         # labels (grayscale PNGs 256x256)
├── LICENSE
├── Makefile            # Makefile for development tasks
├── models
│  └── checkpoints      # saved model checkpoints
├── outputs
│  └── predictions      # predicted masks on test images
├── pyproject.toml      # project configuration, to make it pip-installable
├── README.md
├── requirements.txt
├── src
│  └── dida_roofseg     # main package
│     ├── __init__.py
│     ├── cli.py        # command-line interface
│     ├── dataset.py    # Dataset class for data loading
│     ├── engine.py     # Trainer and Predictor classes
│     ├── io.py         # data I/O utilities
│     ├── model.py      # Model class defining the neural network
│     ├── seed.py       # random seed setting
│     └── viz.py        # visualization utilities
└── tests
   ├── test_cli.py
   ├── test_io.py
   └── test_models.py
```