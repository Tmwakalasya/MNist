# MNIST Dataset Project

This project contains code for working with the MNIST (Modified National Institute of Standards and Technology) dataset, which is a large database of handwritten digits commonly used for training various image processing systems.

## Project Structure

```
MNist/
├── Mnist/
│   └── mnist.py          # Main script for loading and processing MNIST data
├── data/                 # MNIST dataset files (not tracked in git)
│   └── MNIST/
│       └── raw/         # Raw dataset files
└── README.md            # This file
```

## Features

- **Data Loading**: Uses PyTorch and torchvision to load the MNIST dataset
- **Data Preprocessing**: Applies normalization and tensor conversion
- **Batch Processing**: Configurable batch size for training and testing
- **Multi-worker Support**: Uses multiple workers for data loading efficiency

## Requirements

- Python 3.x
- PyTorch
- torchvision

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd MNist
```

2. Install the required packages:
```bash
pip install torch torchvision
```

## Usage

The `mnist.py` file contains code to:
- Download the MNIST dataset automatically
- Apply transformations (normalization and tensor conversion)
- Create data loaders for both training and test sets
- Configure batch processing with multi-worker support

## Dataset

The MNIST dataset will be automatically downloaded to the `data/` directory when you run the script for the first time. This directory is excluded from version control to avoid storing large dataset files in the repository.

## Configuration

You can modify the following parameters in `mnist.py`:
- `batch_size`: Number of samples per batch (default: 4)
- `num_workers`: Number of worker processes for data loading (default: 2)
- Normalization parameters in the transform pipeline

## License

This project is open source and available under the MIT License. 