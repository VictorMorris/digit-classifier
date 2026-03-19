# Digit Classifier

A handwritten digit classifier trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The neural network is written from scratch with just NumPy. It comes with a GUI where you can train a network, watch it learn in real time, and draw your own digits to see if it can figure them out.

With default settings it gets to about 94% accuracy on the test set.

## Screenshots

Set up your hyperparameters (or load a model you already trained):

![Config](img/Screenshot%202026-03-19%20163340.png)

During training, images flash by on the left and the error rate graph updates on the right:

![Training](img/Screenshot%202026-03-19%20163747.png)

When it's done, you get the accuracy and can save the model:

![Save](img/Screenshot%202026-03-19%20163755.png)

Then you can test it on random MNIST images, or draw your own digits on the canvas:

![Prediction](img/Screenshot%202026-03-19%20163822.png)

## Architecture

`784 → 16 → 16 → 10` — 784 inputs (one per pixel), two hidden layers of 16 neurons with sigmoid activation, and 10 outputs (one per digit). Trained with mini-batch SGD and backprop.

## Running it

You'll need Python 3.10+, NumPy, and Matplotlib. Tkinter ships with most Python installs.

Grab the MNIST data files and drop them in `input/`:

```
input/
  train-images.idx3-ubyte
  train-labels.idx1-ubyte
  t10k-images.idx3-ubyte
  t10k-labels.idx1-ubyte
```

Then:

```bash
cd src
python main.py
```

## Project structure

```
src/
  main.py       - GUI and entry point
  network.py    - the neural network
  readdata.py   - MNIST data loader
models/         - saved networks (.npz)
input/          - MNIST data files
```
