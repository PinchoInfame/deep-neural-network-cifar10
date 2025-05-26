# Deep Neural Network for CIFAR-10 Classification

This project implements a deep neural network in MATLAB to classify images from the CIFAR-10 dataset. It includes data loading, preprocessing, network initialization, training with mini-batch gradient descent (with batch normalization), and performance evaluation.

---

## Requirements

- MATLAB (tested with R2024b)
- CIFAR-10 dataset in `.mat` format (download from [here](https://www.cs.toronto.edu/~kriz/cifar.html))
- Basic knowledge of MATLAB scripting and neural networks

---

## Project Structure
<pre> <code> deep-neural-network-cifar10/ ├── mfiles/ │ ├── main.m # Main script to run training and evaluation │ ├── utils/ # Utility functions like LoadBatch.m, InitParam.m, etc. ├── cifar-10-batches-mat/ # CIFAR-10 dataset files (.mat) └── README.md # This file </code> </pre>

---

## Setup Instructions

1. Download and extract the CIFAR-10 dataset in MATLAB format.
2. Place the extracted folder `cifar-10-batches-mat` inside the `cifar-10-matlab` directory.
3. Ensure the folder structure matches the one described above.
4. Open MATLAB and set the working directory to `deep-neural-network-cifar10/mfiles`.
5. Run `main.m` to start training and evaluation.

---

## How It Works

- **Data Loading**: The `LoadBatch` function loads image batches and labels from `.mat` files.
- **Preprocessing**: Data is normalized to have zero mean and unit variance based on the training set statistics.
- **Network Initialization**: The network architecture and parameters are initialized with `InitParam`.
- **Training**: Mini-batch gradient descent with batch normalization is used to train the network. Learning rate scheduling and momentum are implemented.
- **Evaluation**: Training and validation losses, costs, and accuracies are plotted for performance analysis.

---

## Usage

Modify parameters like network size (`m`), regularization (`lambda`), and training epochs (`GDparams.n_epochs`) directly in `main.m` before running.

---

## Results

After training, the script plots:

- Training and validation cost over epochs.
- Training and validation loss over epochs.
- Training and validation accuracy over epochs.

These plots help assess model performance and convergence.

---

## Credits

- This project was developed as part of the coursework **Deep Learning in Data Science** at **KTH Royal Institute of Technology**. The structure, tasks and goals are based on the assignment guidelines provided in the course.
- CIFAR-10 dataset by [Alex Krizhevsky et al.](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## License

MIT License