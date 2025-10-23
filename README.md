# dida-roof-seg
Roof segmentation (DL exercise by Dida)

This task was proposed by Dida as a recruitment exercise. The goal is to segment roofs from aerial images using deep learning techniques.

## Dataset

The dataset consists of 30 aerial images of size 256x256 pixels, along with 25 corresponding labels (referred to as "masks" in this repo) indicating the roof areas. The images are in JPG format, and the masks are in PNG format.  
The data can be found at [https://dida.do/downloads/dida-test-task](https://dida.do/downloads/dida-test-task).

## Instructions from Dida

There are 30 satellite pictures of houses and 25 corresponding labels that indicate the roofs. Take those 25 data points and train a neural network on them — you are completely free about the architecture and are of course allowed to use any predefined version of networks.  
However, you should be able to explain what you are doing — both in terms of code and why certain steps are good choices.  
The preferred language is Python, but you can also use other languages.  
Please evaluate your network on the 5 remaining test images by making predictions of the roofs, and send the predictions together with a short explanation of your approach.

---

## Object-Oriented style

The pipeline follows a lightweight OOP design (Dataset, Model, Trainer, Predictor) to encapsulate data loading, training, and inference logic.  
This makes it easy to extend later (e.g., with augmentation, k-fold CV, or experiment tracking) while keeping the current code minimal and readable.

---

## Project Structure

```bash
dida-roof-seg
.
├── data
│  └── raw
│     ├── images        # input images (PNGs 256x256)
│     └── masks         # labels (grayscale PNGs 256x256)
├── LICENSE
├── Makefile            # Makefile for development tasks
├── models
│  └── checkpoints      # saved model checkpoints
├── outputs
│  └── predictions      # predicted masks on test images
├── pyproject.toml      # project configuration, making it pip-installable
├── README.md
├── requirements.txt
├── src
│  └── dida_roofseg     # main package
│     ├── __init__.py
│     ├── cli.py        # unified command-line interface
│     ├── dataset.py    # Dataset class for data loading
│     ├── engine.py     # Trainer and Predictor classes
│     ├── io.py         # data I/O utilities
│     ├── model.py      # model architecture (encoder + decoder)
│     ├── seed.py       # random seed setting
│     └── viz.py        # visualization utilities
└── tests
   ├── test_cli.py
   ├── test_io.py
   └── test_models.py
```

---

## Python Package and CLI Usage

This repository is structured as a **Python package** (`dida_roofseg`) using a modern `pyproject.toml` build system.  
It can be installed locally in editable mode for development, or directly from GitHub for production use.

### 🧩 Local (development) install

If you’re actively working on the code and want your edits to take effect immediately:

```bash
pip install -e .[dev]
```

This uses `setuptools` in *editable mode*, meaning changes in `src/dida_roofseg/` are reflected instantly.  
The `[dev]` extra installs development dependencies like `pytest`, `ruff`, and `black`.

### 📦 Regular install (end users)

If you just want to use the library or CLI without modifying the source:

```bash
pip install .
```

or directly from GitHub:

```bash
pip install "git+https://github.com/<USER>/<REPO>.git@main"
```

Once installed, the following unified command-line interface becomes available:

```bash
dida-roofseg <subcommand> [options]
```

---

## CLI Reference

### Available subcommands
- **`train`** – Train the roof segmentation model on the 25 labeled images.  
- **`predict`** – Generate roof mask predictions on the 5 unlabeled test images.  

### Common parameters

| Argument | Default | Description |
|-----------|----------|-------------|
| `--data-dir` | `data/raw` | Directory containing input images and masks. |
| `--ckpt-dir` / `--ckpt-path` | `models/checkpoints/` | Path to save or load model checkpoints. |
| `--pred-dir` | `outputs/predictions` | Directory to save predicted masks. |
| `--epochs` | `20` | Number of training epochs. |
| `--batch-size` | `4` | Mini-batch size. |
| `--lr-encoder` | `1e-4` | Learning rate for encoder parameters. |
| `--lr-decoder` | `1e-3` | Learning rate for decoder parameters. |
| `--weight-decay` | `1e-4` | L2 regularization weight. |
| `--freeze-epochs` | `10` | Number of epochs with encoder frozen. |
| `--val-ratio` | `0.2` | Validation split ratio. |
| `--image-size` | `256` | Image resize dimension. |
| `--encoder` | `resnet18` | Encoder backbone (`resnet18`, `resnet34`, or `resnet50`). |
| `--threshold` | `0.5` | Binarization threshold for mask predictions. |
| `--seed` | `42` | Random seed for reproducibility. |
| `--device` | `cpu` | Device to use (`cpu` or `cuda`). |

---

## Example Usage

### Train the model (on GPU if available)
```bash
dida-roofseg train   --data-dir /path/to/data/raw   --device cuda   --epochs 20   --batch-size 4
```

### Predict roof masks on test images
```bash
dida-roofseg predict   --data-dir /path/to/data/raw   --ckpt-path models/checkpoints/best.pth   --device cuda
```

### Programmatic use inside Python or Colab
```python
from dida_roofseg import cli, viz

# Example: run training programmatically
parser = cli.build_parser()
args = parser.parse_args(["train", "--data-dir", "data/raw", "--device", "cuda", "--epochs", "10"])
cli.run_train(args)

# Visualize results
viz.plot_training_curves(history={"train_loss": [1.2, 0.8, 0.6], "val_loss": [1.1, 0.9, 0.7]})
```

---

With this setup, **`dida-roof-seg`** can be used both as a stand-alone package (via `pip install`) and as a reusable Python module inside Jupyter or Colab, while preserving a clean object-oriented structure and a reproducible deep-learning workflow.