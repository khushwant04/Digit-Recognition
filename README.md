# Digit Recognition

Handwritten digit classification on MNIST with a small fully connected neural network in
TensorFlow/Keras. Single-notebook learning project.

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) via `tf.keras.datasets.mnist.load_data()`:
60,000 training and 10,000 test images of 28x28 grayscale digits (downloaded automatically on first
run). Pixel values are scaled to `[0, 1]`.

## Model

```
Flatten(28x28) -> Dense(192, relu) -> Dense(86, relu) -> Dense(10, sigmoid)
```

Compiled with Adam and `sparse_categorical_crossentropy`, trained for 10 epochs on the full
training set. Predictions are taken as `argmax` over the 10 outputs, and the notebook plots a
confusion matrix heatmap with seaborn.

## Results

From the saved outputs in `Digit Recognition.ipynb`:

| Split | Accuracy |
|-------|----------|
| Train (epoch 10) | 0.9940 |
| Test (`model.evaluate`) | 0.9814 |

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook "Digit Recognition.ipynb"
```

Run the cells top to bottom; no model file is saved.

## Notes

- The output layer uses `sigmoid`; `softmax` is the conventional choice for a 10-class
  `sparse_categorical_crossentropy` setup and would give proper class probabilities.
