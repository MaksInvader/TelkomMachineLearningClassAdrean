# Chapter 10: Introduction to Artificial Neural Networks with Keras

## Overview

Artificial Neural Networks (ANNs) are Machine Learning models inspired by biological neurons in our brains. They form the core of Deep Learning and excel at tackling large, complex problems like image classification, speech recognition, and game playing. This chapter covers the fundamentals of ANNs and their implementation using Keras.

## From Biological to Artificial Neurons

### Historical Context

ANNs were first introduced in 1943 by Warren McCulloch and Walter Pitts, who presented a simplified computational model of biological neurons. The field has experienced multiple waves of interest:

- **1960s**: Initial promise unfulfilled, leading to reduced funding
- **1980s**: Revival with new architectures and training techniques
- **1990s**: Overshadowed by techniques like Support Vector Machines
- **Present**: Renewed interest driven by:
  - Huge quantities of training data
  - Increased computing power (Moore's law + GPU acceleration)
  - Improved training algorithms
  - Cloud platform accessibility
  - Theoretical limitations proving benign in practice

### Biological Neurons

A biological neuron consists of:
- **Cell body**: Contains nucleus and complex components
- **Dendrites**: Branching extensions that receive signals
- **Axon**: Long extension that transmits signals
- **Synaptic terminals**: Connected to other neurons' dendrites

Neurons produce electrical impulses (action potentials) that release neurotransmitters. When a neuron receives sufficient neurotransmitters, it fires its own impulses.

### The Perceptron

The Perceptron (1957, Frank Rosenblatt) uses a **Threshold Logic Unit (TLU)**, which:
- Takes numerical inputs and outputs
- Assigns weights to each input connection
- Computes weighted sum: `z = w₁x₁ + w₂x₂ + ... + wₙxₙ = x⊺w`
- Applies step function: `h_w(x) = step(z)`

**Common step functions:**

```
Equation 10-1: Step functions used in Perceptrons

heaviside(z) = {0 if z < 0
               {1 if z ≥ 0

sgn(z) = {-1 if z < 0
         { 0 if z = 0
         {+1 if z > 0
```

**Perceptron architecture:**
- Single layer of TLUs
- Each TLU connected to all inputs (fully connected/dense layer)
- Input neurons are passthrough
- Includes bias neuron (outputs 1)

**Computing layer outputs:**

```
Equation 10-2: Computing outputs of a fully connected layer

h_W,b(X) = φ(XW + b)

Where:
- X: input features matrix (one row per instance)
- W: weight matrix (one row per input neuron, one column per artificial neuron)
- b: bias vector (one bias term per artificial neuron)
- φ: activation function
```

**Perceptron Learning Rule:**

```
Equation 10-3: Perceptron learning rule (weight update)

w_i,j(next step) = w_i,j + η(ŷ_j - y_j)x_i

Where:
- w_i,j: connection weight between ith input and jth output neuron
- x_i: ith input value
- y_j: output of jth neuron
- ŷ_j: target output
- η: learning rate
```

**Perceptron Convergence Theorem**: If training instances are linearly separable, the algorithm converges to a solution.

**Scikit-Learn implementation:**

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length, petal width
y = (iris.target == 0).astype(np.int)  # Iris setosa?

per_clf = Perceptron()
per_clf.fit(X, y)
y_pred = per_clf.predict([[2, 0.5]])
```

**Limitations**: Perceptrons cannot solve non-linearly separable problems (e.g., XOR).

### The Multilayer Perceptron (MLP)

MLPs overcome Perceptron limitations by stacking multiple layers:
- **Input layer**: Passthrough neurons
- **Hidden layers**: One or more layers of TLUs
- **Output layer**: Final layer of TLUs
- **Bias neurons**: Included in each layer except output

Architecture terminology:
- **Lower layers**: Close to input
- **Upper layers**: Close to output
- **Deep Neural Network (DNN)**: ANN with deep stack of hidden layers
- **Feedforward Neural Network (FNN)**: Signal flows only forward

### Backpropagation

Introduced in 1986 by Rumelhart, Hinton, and Williams, backpropagation is Gradient Descent using efficient automatic differentiation (autodiff).

**Algorithm steps:**

1. **Forward pass**: Process mini-batch through all layers, preserving intermediate results
2. **Measure error**: Compute loss function on network output
3. **Backward pass**: Apply chain rule to measure error contribution from each connection (reverse-mode autodiff)
4. **Gradient Descent step**: Tweak connection weights to reduce error

**Key requirements:**
- Random weight initialization (to break symmetry)
- Replace step function with differentiable activation function

**Popular activation functions:**

1. **Logistic (Sigmoid)**: `σ(z) = 1 / (1 + exp(-z))`
   - Output range: [0, 1]
   - S-shaped, continuous, differentiable

2. **Hyperbolic tangent**: `tanh(z) = 2σ(2z) - 1`
   - Output range: [-1, 1]
   - Helps speed up convergence

3. **ReLU (Rectified Linear Unit)**: `ReLU(z) = max(0, z)`
   - Most common default
   - Fast to compute
   - No maximum output value (reduces gradient issues)
   - Not differentiable at z=0, derivative is 0 for z<0 (works well in practice)

**Why activation functions?** Without nonlinearity, stacking layers yields only linear transformations. Nonlinear activations enable learning complex patterns.

## MLP Architectures

### Regression MLPs

**Architecture guidelines:**

| Hyperparameter | Typical Value |
|----------------|---------------|
| # input neurons | One per input feature |
| # hidden layers | 1 to 5 |
| # neurons per hidden layer | 10 to 100 |
| # output neurons | 1 per prediction dimension |
| Hidden activation | ReLU (or SELU) |
| Output activation | None, or ReLU/softplus (positive), or logistic/tanh (bounded) |
| Loss function | MSE or MAE/Huber (if outliers) |

**Huber loss**: Quadratic when error < δ (typically 1), linear when error > δ. Less sensitive to outliers than MSE, converges faster than MAE.

### Classification MLPs

**Binary classification:**
- Single output neuron with logistic activation
- Output: estimated probability of positive class

**Multilabel binary classification:**
- One output neuron per label (logistic activation)
- Probabilities don't sum to 1
- Example: spam/ham + urgent/non-urgent email

**Multiclass classification:**
- One output neuron per class
- Softmax activation on entire output layer
- Probabilities sum to 1
- Loss: cross-entropy (log loss)

**Architecture summary:**

| Hyperparameter | Binary | Multilabel Binary | Multiclass |
|----------------|--------|-------------------|------------|
| Input/hidden layers | Same as regression | Same as regression | Same as regression |
| # output neurons | 1 | 1 per label | 1 per class |
| Output activation | Logistic | Logistic | Softmax |
| Loss function | Cross entropy | Cross entropy | Cross entropy |

## Implementing MLPs with Keras

### Installation

```bash
$ cd $ML_PATH
$ source my_env/bin/activate  # Linux/macOS
$ .\my_env\Scripts\activate   # Windows
$ python3 -m pip install -U tensorflow
```

**Testing installation:**

```python
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)      # '2.0.0'
print(keras.__version__)   # '2.2.4-tf'
```

### Building Image Classifier with Sequential API

**Loading Fashion MNIST:**

```python
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Create validation set and scale features
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

**Creating model:**

```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
```

**Layer explanations:**
- `Flatten`: Converts 2D images to 1D arrays (preprocessing, no parameters)
- `Dense`: Fully connected layer managing weights and biases
- `input_shape=[28, 28]`: Specifies input dimensions (excluding batch size)

**Model summary:**

```python
model.summary()
# Shows layers, output shapes, and parameter counts
# Total params: 266,610 (784×300 + 300 + 300×100 + 100 + 100×10 + 10)
```

**Accessing layers and weights:**

```python
hidden1 = model.layers[1]
weights, biases = hidden1.get_weights()
# weights.shape: (784, 300)
# biases.shape: (300,)
```

**Compiling model:**

```python
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
```

- `sparse_categorical_crossentropy`: For sparse labels (class indices)
- Use `categorical_crossentropy` for one-hot vectors
- Use `binary_crossentropy` + sigmoid for binary classification

**Training:**

```python
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
```

**Evaluation:**

```python
model.evaluate(X_test, y_test)  # Returns [loss, accuracy]
```

**Making predictions:**

```python
X_new = X_test[:3]
y_proba = model.predict(X_new)  # Probability per class
y_pred = model.predict_classes(X_new)  # Predicted class
```

### Building Regression MLP

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
```

**Key differences from classification:**
- Output layer: Single neuron, no activation
- Loss: MSE (or MAE/Huber if outliers)

## Building Complex Models with Functional API

### Wide & Deep Neural Network

Connects inputs directly to output layer alongside deep path, learning both deep patterns and simple rules.

```python
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])
```

**Functional API characteristics:**
- Specify connections by calling layers as functions
- No data processing during model building
- More flexible than Sequential API

### Multiple Inputs

```python
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.Model(inputs=[input_A, input_B], outputs=[output])

# Training with multiple inputs
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                    validation_data=((X_valid_A, X_valid_B), y_valid))
```

### Multiple Outputs

**Use cases:**
- Task demands it (e.g., object localization + classification)
- Multiple independent tasks (better results than separate networks)
- Regularization technique (auxiliary outputs)

```python
output = keras.layers.Dense(1, name="main_output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.Model(inputs=[input_A, input_B], 
                    outputs=[output, aux_output])

model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], 
              optimizer="sgd")
history = model.fit([X_train_A, X_train_B], [y_train, y_train], 
                    epochs=20,
                    validation_data=([X_valid_A, X_valid_B], 
                                    [y_valid, y_valid]))
```

## Subclassing API for Dynamic Models

For loops, conditional branching, and dynamic behaviors:

```python
class WideAndDeepModel(keras.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
    
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel()
```

**Trade-offs:**
- **Pros**: Maximum flexibility, great for research
- **Cons**: Architecture hidden in `call()`, harder to inspect/save/clone, easier to make mistakes

## Saving and Restoring Models

```python
model.save("my_keras_model.h5")  # Saves architecture, parameters, optimizer
model = keras.models.load_model("my_keras_model.h5")
```

**Note**: Full save/load works with Sequential and Functional APIs, not Subclassing API (use `save_weights()`/`load_weights()` for Subclassing).

## Using Callbacks

Callbacks allow Keras to call functions at various training stages.

### ModelCheckpoint

```python
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5",
                                                save_best_only=True)
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb])
```

### EarlyStopping

```python
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

### Custom Callbacks

```python
class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))
```

**Available callback methods:**
- Training: `on_train_begin/end`, `on_epoch_begin/end`, `on_batch_begin/end`
- Evaluation: `on_test_begin/end`, `on_test_batch_begin/end`
- Prediction: `on_predict_begin/end`, `on_predict_batch_begin/end`

## Using TensorBoard for Visualization

TensorBoard provides interactive visualization of learning curves, computation graphs, training statistics, and more.

**Setup:**

```python
import os

root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
```

**Using TensorBoard callback:**

```python
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tensorboard_cb])
```

**Starting TensorBoard server:**

```bash
$ tensorboard --logdir=./my_logs --port=6006
```

Or in Jupyter:

```python
%load_ext tensorboard
%tensorboard --logdir=./my_logs --port=6006
```

**Lower-level API for custom logging:**

```python
test_logdir = get_run_logdir()
writer = tf.summary.create_file_writer(test_logdir)
with writer.as_default():
    for step in range(1, 1001):
        tf.summary.scalar("my_scalar", np.sin(step / 10), step=step)
        data = (np.random.randn(100) + 2) * step / 100
        tf.summary.histogram("my_hist", data, buckets=50, step=step)
        images = np.random.rand(2, 32, 32, 3)
        tf.summary.image("my_images", images * step / 1000, step=step)
        texts = ["The step is " + str(step), "Its square is " + str(step**2)]
        tf.summary.text("my_text", texts, step=step)
        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step)
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1])
        tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)
```

## Fine-Tuning Hyperparameters

### Using GridSearchCV/RandomizedSearchCV

```python
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

print(rnd_search_cv.best_params_)
print(rnd_search_cv.best_score_)
model = rnd_search_cv.best_estimator_.model
```

### Advanced Hyperparameter Optimization Libraries

- **Hyperopt**: Optimizes complex search spaces
- **Hyperas, kopt, Talos**: Keras-specific (based on Hyperopt)
- **Keras Tuner**: Google's easy-to-use library for Keras
- **Scikit-Optimize**: Bayesian optimization with `BayesSearchCV`
- **Spearmint**: Bayesian optimization library
- **Hyperband**: Fast tuning based on bandit algorithms
- **Sklearn-Deap**: Evolutionary algorithms

Cloud services: Google Cloud AI Platform, Arimo, SigOpt, CallDesk's Oscar

### Number of Hidden Layers

**Guidelines:**
- Start with 1-2 hidden layers for many problems
- Single hidden layer can theoretically model complex functions but needs many neurons
- Deep networks are more parameter-efficient for complex problems
- Hierarchical structure: lower layers model low-level features, higher layers combine them
- Benefits: faster convergence, better generalization, enables transfer learning
- Very complex tasks (large image classification, speech recognition) need dozens/hundreds of layers
- Increase layers until overfitting begins

**Transfer learning**: Reuse lower layers from pretrained models to kickstart training on new tasks.

### Number of Neurons per Hidden Layer

**Guidelines:**
- Input/output neurons determined by task
- Historical approach: pyramid (fewer neurons in higher layers)
- Modern approach: same number in all hidden layers (simpler, works as well)
- First hidden layer can sometimes be bigger
- **"Stretch pants" approach**: Use more neurons than needed, then apply regularization
- Avoid bottleneck layers (too few neurons lose information)
- **General principle**: More layers > more neurons per layer

### Learning Rate, Batch Size, and Other Hyperparameters

**Learning rate:**
- Most important hyperparameter
- Optimal ≈ half of maximum (where training diverges)
- **Finding optimal learning rate:**
  1. Train for few hundred iterations
  2. Start with very low rate (10⁻⁵)
  3. Gradually increase to high value (10)
  4. Plot loss vs learning rate
  5. Choose rate ~10× lower than turning point (where loss shoots up)

**Optimizer:**
- SGD is basic; better optimizers available (Chapter 11)
- Always retune learning rate when changing optimizer

**Batch size:**
- Large batches: efficient GPU processing, more instances/second
- Small batches (2-32): often better generalization, shorter training time
- Large batches (up to 8,192): possible with techniques like learning rate warmup
- Strategy: Try large batch with warmup; if unstable/poor performance, use small batch
- Optimal learning rate depends on batch size

**Activation function:**
- Hidden layers: ReLU (default)
- Output layer: depends on task

**Number of iterations:**
- Use early stopping instead of tuning directly

**Important**: When modifying any hyperparameter, update learning rate accordingly.

## Key Takeaways

1. **ANNs are inspired by biological neurons** but have evolved beyond strict biological analogies
2. **Perceptrons** are limited to linear classification; MLPs overcome this
3. **Backpropagation** efficiently computes gradients using reverse-mode autodiff
4. **Keras provides three APIs**:
   - Sequential: Simple, single stack of layers
   - Functional: More flexible, supports complex architectures
   - Subclassing: Maximum flexibility for dynamic models
5. **Model lifecycle**: Build → Compile → Train → Evaluate → Predict
6. **Callbacks** enable checkpointing, early stopping, and custom behavior
7. **TensorBoard** provides powerful visualization capabilities
8. **Hyperparameter tuning** is crucial; use randomized search or advanced optimization libraries
9. **Architecture guidelines**:
   - Deep networks more efficient than wide shallow ones
   - Start with 1-2 hidden layers, increase until overfitting
   - Use "stretch pants" approach: more neurons + regularization
10. **Learning rate** is the most critical hyperparameter to tune

## Next Steps

Future chapters will cover:
- Training very deep networks
- Customizing models with TensorFlow's lower-level API
- Efficient data loading with Data API
- Convolutional Neural Networks (CNNs) for images
- Recurrent Neural Networks (RNNs) for sequences
- Autoencoders for representation learning
- Generative Adversarial Networks (GANs)
