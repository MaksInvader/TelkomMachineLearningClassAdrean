# Chapter 15: Processing Sequences Using RNNs and CNNs

## Overview

Recurrent Neural Networks (RNNs) are designed to predict future values and process sequences of arbitrary lengths. They excel at analyzing time series data, natural language processing, and any task involving sequential patterns. This chapter covers:

- Fundamental RNN concepts and training via backpropagation through time
- Time series forecasting techniques
- Handling unstable gradients and limited short-term memory
- Advanced architectures: LSTM, GRU, and WaveNet

## Recurrent Neurons and Layers

### Basic Architecture

Unlike feedforward networks, RNNs have connections pointing backward, allowing them to maintain state across time steps. A recurrent neuron receives:
- Current input **x(t)**
- Its own output from the previous time step **y(t-1)**

At the first time step, the previous output is set to 0.

### Mathematical Formulation

For a single instance:

**Equation 15-1: Output of a recurrent layer**
```
y(t) = φ(Wx · x(t) + Wy · y(t-1) + b)
```

For a mini-batch:

**Equation 15-2: Outputs for all instances**
```
Y(t) = φ(X(t)Wx + Y(t-1)Wy + b)
     = φ([X(t) Y(t-1)]W + b)
```

Where:
- **Y(t)**: m × n_neurons matrix (outputs at time step t)
- **X(t)**: m × n_inputs matrix (inputs for all instances)
- **Wx**: n_inputs × n_neurons weight matrix
- **Wy**: n_neurons × n_neurons weight matrix
- **b**: bias vector of size n_neurons
- **W**: concatenated weight matrix of shape (n_inputs + n_neurons) × n_neurons

### Memory Cells

A memory cell preserves state across time steps. The cell's state at time step t is:
```
h(t) = f(h(t-1), x(t))
```

In basic cells, output equals state: **y(t) = h(t)**

### Input and Output Sequences

RNNs support four main architectures:

1. **Sequence-to-Sequence**: Input sequence → Output sequence (e.g., stock price prediction)
2. **Sequence-to-Vector**: Input sequence → Single output (e.g., sentiment analysis)
3. **Vector-to-Sequence**: Single input → Output sequence (e.g., image captioning)
4. **Encoder-Decoder**: Sequence → Vector → Sequence (e.g., machine translation)

## Training RNNs

### Backpropagation Through Time (BPTT)

Training process:
1. Unroll the RNN through time
2. Forward pass through the unrolled network
3. Evaluate output sequence using cost function **C(Y(0), Y(1), ..., Y(T))**
4. Backpropagate gradients through all time steps
5. Update parameters using computed gradients

The same parameters **W** and **b** are used at each time step, so backpropagation sums gradients over all time steps.

## Forecasting Time Series

### Data Preparation

Time series are represented as 3D arrays: **[batch_size, time_steps, dimensionality]**

Example data generation:

```python
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))  # wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)  # noise
    return series[..., np.newaxis].astype(np.float32)
```

Creating datasets:

```python
n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
```

### Baseline Metrics

**Naive forecasting**: Predict the last value (MSE ≈ 0.020)

**Linear model**: Flatten inputs, apply Dense layer (MSE ≈ 0.004)

```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50, 1]),
    keras.layers.Dense(1)
])
```

### Simple RNN Implementation

```python
model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1])
])
```

By default, SimpleRNN:
- Uses hyperbolic tangent activation
- Returns only the final output (set `return_sequences=True` for all outputs)
- Has fewer parameters than linear models but may underperform initially (MSE ≈ 0.014)

### Deep RNNs

Stack multiple recurrent layers for better performance:

```python
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.SimpleRNN(1)
])
```

**Important**: Set `return_sequences=True` for all layers except the last to output 3D arrays.

Improved architecture with Dense output layer:

```python
model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(1)
])
```

This converges faster and allows flexible output activation functions (MSE ≈ 0.003).

### Forecasting Multiple Steps Ahead

**Option 1: Iterative prediction**
Predict one step at a time, feeding predictions back as inputs. Less accurate for longer horizons but simple.

**Option 2: Direct multi-step prediction**
Train model to predict all future values at once:

```python
series = generate_time_series(10000, n_steps + 10)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10)
])
```

**Option 3: Sequence-to-sequence prediction**
Predict next values at every time step:

```python
# Prepare targets
Y = np.empty((10000, n_steps, 10))
for step_ahead in range(1, 11):
    Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
```

The `TimeDistributed` layer applies the Dense layer at every time step efficiently.

Custom metric for last time step evaluation:

```python
def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
```

This achieves MSE ≈ 0.006, 25% better than direct multi-step prediction.

## Handling Long Sequences

### Fighting Unstable Gradients

**Challenges**:
- Gradients can explode or vanish over many time steps
- Nonsaturating activations (ReLU) may worsen instability
- Batch Normalization works poorly between time steps

**Solutions**:

1. **Use saturating activations**: tanh (default for RNNs) prevents runaway outputs
2. **Gradient Clipping**: Monitor and clip gradient magnitudes
3. **Layer Normalization**: More effective than Batch Normalization for RNNs

**Custom cell with Layer Normalization**:

```python
class LNSimpleRNNCell(keras.layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = keras.layers.SimpleRNNCell(units, activation=None)
        self.layer_norm = keras.layers.LayerNormalization()
        self.activation = keras.activations.get(activation)
    
    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]

model = keras.models.Sequential([
    keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True, 
                     input_shape=[None, 1]),
    keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
```

4. **Dropout**: Use `dropout` and `recurrent_dropout` hyperparameters in Keras recurrent layers

### Tackling Short-Term Memory

Basic RNNs lose information over long sequences (typically ~10 steps). Advanced cells extend memory significantly.

## LSTM Cells

**Long Short-Term Memory** (LSTM) cells maintain both short-term and long-term state:

- **h(t)**: Short-term state
- **c(t)**: Long-term state

### Architecture Components

1. **Main layer (g(t))**: Analyzes current input and previous state
2. **Forget gate (f(t))**: Controls what to erase from long-term state
3. **Input gate (i(t))**: Controls what to add to long-term state
4. **Output gate (o(t))**: Controls what to read from long-term state

### LSTM Computations

**Equation 15-3: LSTM equations**

```
i(t) = σ(Wxi · x(t) + Whi · h(t-1) + bi)
f(t) = σ(Wxf · x(t) + Whf · h(t-1) + bf)
o(t) = σ(Wxo · x(t) + Who · h(t-1) + bo)
g(t) = tanh(Wxg · x(t) + Whg · h(t-1) + bg)
c(t) = f(t) ⊗ c(t-1) + i(t) ⊗ g(t)
y(t) = h(t) = o(t) ⊗ tanh(c(t))
```

Where σ is the sigmoid function and ⊗ is element-wise multiplication.

**Note**: TensorFlow initializes **bf** to all 1s (instead of 0s) to prevent forgetting everything initially.

### Implementation

```python
model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
```

Or using RNN layer with LSTMCell:

```python
model = keras.models.Sequential([
    keras.layers.RNN(keras.layers.LSTMCell(20), return_sequences=True,
                     input_shape=[None, 1]),
    keras.layers.RNN(keras.layers.LSTMCell(20), return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
```

**Prefer the LSTM layer**: It uses optimized GPU implementations.

### Peephole Connections

Peephole LSTMs let gate controllers peek at long-term state:
- Forget and input gates see **c(t-1)**
- Output gate sees **c(t)**

Use `tf.keras.experimental.PeepholeLSTMCell` for this variant.

## GRU Cells

**Gated Recurrent Unit** (GRU) is a simplified LSTM with similar performance:

### Simplifications

1. **Merged state**: Single vector **h(t)** (no separate long-term state)
2. **Combined gates**: Single gate **z(t)** controls both forget and input
3. **No output gate**: Full state is always output
4. **Reset gate r(t)**: Controls which part of previous state is shown

### GRU Computations

**Equation 15-4: GRU equations**

```
z(t) = σ(Wxz · x(t) + Whz · h(t-1) + bz)
r(t) = σ(Wxr · x(t) + Whr · h(t-1) + br)
g(t) = tanh(Wxg · x(t) + Whg · (r(t) ⊗ h(t-1)) + bg)
h(t) = z(t) ⊗ h(t-1) + (1 - z(t)) ⊗ g(t)
```

### Implementation

```python
model = keras.models.Sequential([
    keras.layers.GRU(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])
```

GRUs perform similarly to LSTMs with fewer parameters and faster training.

## Using 1D Convolutional Layers

### Motivation

1D CNNs can help RNNs by:
- Shortening sequences (enabling detection of longer patterns)
- Learning short sequential patterns efficiently
- Reducing computational burden on recurrent layers

### Architecture

A 1D convolutional layer slides kernels across sequences, producing feature maps. Each kernel detects specific short patterns.

### Example with Downsampling

```python
model = keras.models.Sequential([
    keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
                       input_shape=[None, 1]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

# Adjust targets for stride=2 and kernel_size=4
history = model.fit(X_train, Y_train[:, 3::2], epochs=20,
                   validation_data=(X_valid, Y_valid[:, 3::2]))
```

This model achieves better performance by preprocessing sequences with convolution.

## WaveNet Architecture

WaveNet uses stacked 1D convolutional layers with **exponentially increasing dilation rates** to efficiently process very long sequences.

### Key Features

- **Dilation rates double at each layer**: 1, 2, 4, 8, 16, 32, ...
- **Causal padding**: No peeking into the future
- **Efficient receptive field**: A stack of 10 layers with these rates has a receptive field of 1,024 time steps
- **Multi-scale learning**: Lower layers learn short-term patterns, higher layers learn long-term patterns

### Implementation

```python
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[None, 1]))

# Stack convolutional layers with increasing dilation
for rate in (1, 2, 4, 8) * 2:
    model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal",
                                  activation="relu", dilation_rate=rate))

# Output layer
model.add(keras.layers.Conv1D(filters=10, kernel_size=1))

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=20,
                   validation_data=(X_valid, Y_valid))
```

**"Causal" padding**: Ensures the model doesn't peek into future time steps (equivalent to left-padding with zeros).

### Performance

WaveNet achieved state-of-the-art results on:
- Text-to-speech synthesis
- Audio generation
- Music composition

It can handle sequences with tens of thousands of time steps, which LSTMs and GRUs struggle with.

## Key Takeaways

1. **RNNs are powerful for sequences** but face challenges with long-term dependencies
2. **Simple RNNs** are easy to implement but limited to ~10 time steps
3. **Deep RNNs** improve performance by stacking layers
4. **LSTM and GRU cells** extend memory to ~100 time steps through gating mechanisms
5. **1D CNNs** complement RNNs by preprocessing sequences
6. **WaveNet** demonstrates that pure convolutional architectures can handle very long sequences efficiently
7. **Layer Normalization** is more effective than Batch Normalization for RNNs
8. **Sequence-to-sequence models** with predictions at every time step train faster and more stably

## Exercises

1. Applications for different RNN types: seq-to-seq (translation), seq-to-vector (classification), vector-to-seq (image captioning)
2. RNN inputs: 3D arrays [batch_size, time_steps, features]
3. Deep seq-to-seq: all layers need `return_sequences=True`; seq-to-vector: only intermediate layers
4. Forecasting 7 days: Use sequence-to-sequence RNN or predict all 7 values at once
5. Main difficulties: unstable gradients (use gradient clipping, Layer Normalization), limited memory (use LSTM/GRU)
6. LSTM architecture: 4 layers (main + 3 gates), long-term and short-term states
7. 1D CNNs in RNNs: Shorten sequences, learn local patterns efficiently
8. Video classification: 3D CNN or CNN + RNN (frame features → sequence classification)
9-10. Practical projects with SketchRNN and Bach chorales datasets