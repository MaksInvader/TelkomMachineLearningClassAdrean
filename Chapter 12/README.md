# Chapter 12: Custom Models and Training with TensorFlow

## Overview

This chapter explores TensorFlow's lower-level Python API, enabling you to build custom loss functions, metrics, layers, models, initializers, regularizers, weight constraints, and custom training loops. While 95% of use cases only require tf.keras, understanding these advanced techniques provides the flexibility needed for specialized architectures and training procedures.

## TensorFlow Quick Tour

**What TensorFlow Offers:**
- NumPy-like core with GPU support
- Distributed computing across devices and servers
- Just-in-time (JIT) compiler for optimization
- Portable computation graphs
- Autodiff with excellent optimizers (RMSProp, Nadam)
- Rich ecosystem: TensorBoard, TensorFlow Extended (TFX), TensorFlow Hub, model garden

**Architecture:**
- High-level APIs: tf.keras, tf.data
- Lower-level Python API for direct tensor manipulation
- Efficient C++ execution engine with GPU/TPU support
- Cross-platform: Windows, Linux, macOS, mobile (TensorFlow Lite), web (TensorFlow.js)

## Using TensorFlow like NumPy

### Tensors and Operations

Create tensors with `tf.constant()`:

```python
>>> tf.constant([[1., 2., 3.], [4., 5., 6.]])  # matrix
<tf.Tensor: id=0, shape=(2, 3), dtype=float32, numpy=
array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)>

>>> tf.constant(42)  # scalar
<tf.Tensor: id=1, shape=(), dtype=int32, numpy=42>
```

**Tensor properties:**
```python
>>> t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
>>> t.shape
TensorShape([2, 3])
>>> t.dtype
tf.float32
```

**Indexing (NumPy-like):**
```python
>>> t[:, 1:]
<tf.Tensor: id=5, shape=(2, 2), dtype=float32, numpy=
array([[2., 3.],
       [5., 6.]], dtype=float32)>
```

**Operations:**
```python
>>> t + 10  # equivalent to tf.add(t, 10)
>>> tf.square(t)
>>> t @ tf.transpose(t)  # matrix multiplication
```

**Common operations:** `tf.add()`, `tf.multiply()`, `tf.square()`, `tf.exp()`, `tf.sqrt()`, `tf.reshape()`, `tf.squeeze()`, `tf.tile()`, `tf.reduce_mean()`, `tf.reduce_sum()`, `tf.reduce_max()`

### Tensors and NumPy

Seamless interoperability:

```python
>>> a = np.array([2., 4., 5.])
>>> tf.constant(a)  # NumPy to tensor
>>> t.numpy()  # or np.array(t) - tensor to NumPy
>>> tf.square(a)  # TensorFlow ops on NumPy arrays
>>> np.square(t)  # NumPy ops on tensors
```

**Important:** NumPy uses 64-bit precision by default, TensorFlow uses 32-bit. Always set `dtype=tf.float32` when creating tensors from NumPy arrays.

### Type Conversions

TensorFlow does NOT perform automatic type conversions:

```python
>>> tf.constant(2.) + tf.constant(40)
# InvalidArgumentError: expected to be a float

>>> tf.constant(2.) + tf.constant(40., dtype=tf.float64)
# InvalidArgumentError: expected to be a double
```

Use `tf.cast()` for explicit conversion:

```python
>>> t2 = tf.constant(40., dtype=tf.float64)
>>> tf.constant(2.0) + tf.cast(t2, tf.float32)
<tf.Tensor: id=136, shape=(), dtype=float32, numpy=42.0>
```

### Variables

Mutable tensors for weights and parameters:

```python
>>> v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
>>> v.assign(2 * v)  # => [[2., 4., 6.], [8., 10., 12.]]
>>> v[0, 1].assign(42)  # => [[2., 42., 6.], [8., 10., 12.]]
>>> v[:, 2].assign([0., 1.])  # => [[2., 42., 0.], [8., 10., 1.]]
>>> v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.])
```

**Note:** Keras `add_weight()` method handles variable creation automatically. Optimizers update variables directly.

### Other Data Structures

- **Sparse tensors** (`tf.SparseTensor`): For tensors with mostly zeros
- **Tensor arrays** (`tf.TensorArray`): Lists of tensors with fixed/dynamic size
- **Ragged tensors** (`tf.RaggedTensor`): Lists of lists with varying lengths
- **String tensors**: Byte strings (`tf.string`) or Unicode (`tf.int32` code points)
- **Sets**: Regular/sparse tensors (tf.sets package)
- **Queues**: FIFO, priority, shuffle, padding queues (tf.queue package)

## Customizing Models and Training Algorithms

### Custom Loss Functions

**Simple function approach:**

```python
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

model.compile(loss=huber_fn, optimizer="nadam")
model.fit(X_train, y_train, [...])
```

**Loading models with custom functions:**

```python
model = keras.models.load_model("my_model_with_a_custom_loss.h5",
                                custom_objects={"huber_fn": huber_fn})
```

**Configurable loss with closure:**

```python
def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn

model.compile(loss=create_huber(2.0), optimizer="nadam")
```

**Class-based approach (saves hyperparameters):**

```python
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
    
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}

model.compile(loss=HuberLoss(2.), optimizer="nadam")
```

### Custom Activation Functions, Initializers, Regularizers, Constraints

**Function-based examples:**

```python
def my_softplus(z):
    return tf.math.log(tf.exp(z) + 1.0)

def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)

# Usage
layer = keras.layers.Dense(30, activation=my_softplus,
                           kernel_initializer=my_glorot_initializer,
                           kernel_regularizer=my_l1_regularizer,
                           kernel_constraint=my_positive_weights)
```

**Class-based regularizer (saves hyperparameters):**

```python
class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    
    def get_config(self):
        return {"factor": self.factor}
```

### Custom Metrics

**Simple function (averaged over batches):**

```python
model.compile(loss="mse", optimizer="nadam", metrics=[create_huber(2.0)])
```

**Streaming metric (maintains state across batches):**

```python
>>> precision = keras.metrics.Precision()
>>> precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
<tf.Tensor: numpy=0.8>
>>> precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])
<tf.Tensor: numpy=0.5>  # Overall precision, not just second batch

>>> precision.result()  # Get current value
>>> precision.variables  # View internal state
>>> precision.reset_states()  # Reset for new epoch
```

**Custom streaming metric:**

```python
class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    
    def result(self):
        return self.total / self.count
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}
```

### Custom Layers

**Stateless layer (Lambda):**

```python
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
```

**Stateful layer (with weights):**

```python
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", 
            shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal")
        self.bias = self.add_weight(
            name="bias", 
            shape=[self.units], 
            initializer="zeros")
        super().build(batch_input_shape)
    
    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)
    
    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}
```

**Layer with training behavior:**

```python
class MyGaussianNoise(keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
    
    def call(self, X, training=None):
        if training:
            noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
            return X + noise
        else:
            return X
    
    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape
```

### Custom Models

**Residual block layer:**

```python
class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons, activation="elu",
                                         kernel_initializer="he_normal")
                      for _ in range(n_layers)]
    
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z  # Skip connection
```

**Custom model using Subclassing API:**

```python
class ResidualRegressor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(30, activation="elu",
                                         kernel_initializer="he_normal")
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = keras.layers.Dense(output_dim)
    
    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)
```

### Losses Based on Model Internals

**Model with reconstruction loss:**

```python
class ReconstructingRegressor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(30, activation="selu",
                                         kernel_initializer="lecun_normal")
                      for _ in range(5)]
        self.out = keras.layers.Dense(output_dim)
    
    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs)
        super().build(batch_input_shape)
    
    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)  # Add regularization loss
        return self.out(Z)
```

## Computing Gradients Using Autodiff

**Basic gradient computation:**

```python
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2

w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
# Returns: [<tf.Tensor: numpy=36.0>, <tf.Tensor: numpy=10.0>]
```

**Persistent tape (for multiple gradient calls):**

```python
with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1)
dz_dw2 = tape.gradient(z, w2)
del tape  # Free resources
```

**Watching non-variable tensors:**

```python
c1, c2 = tf.constant(5.), tf.constant(3.)
with tf.GradientTape() as tape:
    tape.watch(c1)
    tape.watch(c2)
    z = f(c1, c2)

gradients = tape.gradient(z, [c1, c2])
```

**Stopping gradients:**

```python
def f(w1, w2):
    return 3 * w1 ** 2 + tf.stop_gradient(2 * w1 * w2)

with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])  # => [tensor 30., None]
```

**Custom gradients (for numerical stability):**

```python
@tf.custom_gradient
def my_better_softplus(z):
    exp = tf.exp(z)
    def my_softplus_gradients(grad):
        return grad / (1 + 1 / exp)
    return tf.math.log(exp + 1), my_softplus_gradients
```

## Custom Training Loops

**Setup:**

```python
l2_reg = keras.regularizers.l2(0.05)
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal",
                      kernel_regularizer=l2_reg),
    keras.layers.Dense(1, kernel_regularizer=l2_reg)
])

def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)
    return X[idx], y[idx]

n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size
optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.mean_squared_error
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.MeanAbsoluteError()]
```

**Training loop:**

```python
for epoch in range(1, n_epochs + 1):
    print("Epoch {}/{}".format(epoch, n_epochs))
    for step in range(1, n_steps + 1):
        X_batch, y_batch = random_batch(X_train_scaled, y_train)
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        mean_loss(loss)
        for metric in metrics:
            metric(y_batch, y_pred)
        print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
    
    print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
    for metric in [mean_loss] + metrics:
        metric.reset_states()
```

**Applying weight constraints:**

```python
for variable in model.variables:
    if variable.constraint is not None:
        variable.assign(variable.constraint(variable))
```

## TensorFlow Functions and Graphs

**Converting Python functions to TF Functions:**

```python
def cube(x):
    return x ** 3

# Method 1: Explicit conversion
tf_cube = tf.function(cube)

# Method 2: Decorator (preferred)
@tf.function
def tf_cube(x):
    return x ** 3

>>> tf_cube(tf.constant(2.0))
<tf.Tensor: shape=(), dtype=float32, numpy=8.0>
```

**Benefits:**
- Automatic graph generation and optimization
- Much faster execution (pruning, simplification, parallelization)
- Keras automatically converts custom functions to TF Functions

**Polymorphism:**
- New graph generated for each unique set of input shapes/dtypes
- Same graph reused for tensor arguments with matching shape/dtype
- Numerical Python values generate new graphs for each distinct value

### AutoGraph and Tracing

**Process:**
1. **AutoGraph**: Analyzes source code, replaces control flow with TensorFlow operations
2. **Tracing**: Calls upgraded function with symbolic tensors (no actual values)
3. **Graph Generation**: Operations add nodes representing themselves and outputs

### TF Function Rules

1. **External libraries** run only during tracing (not in graph)
   - Use TensorFlow equivalents: `tf.reduce_sum()` vs `np.sum()`
   - Random operations: `tf.random.uniform()` vs `np.random.rand()`
   - Side effects only occur during tracing
   - Use `tf.py_function()` for arbitrary Python (reduces performance/portability)

2. **Calling other functions**: They follow same rules, no decorator needed

3. **Stateful objects**: Create variables/datasets only on first call
   - Prefer creating variables outside TF Function
   - Use `variable.assign()` not `=` operator

4. **Source code availability**: Required for graph generation

5. **Loop iteration**: Use `for i in tf.range(x)` not `for i in range(x)`

6. **Vectorization**: Prefer vectorized implementations over loops

**Dynamic models:**
```python
# Disable automatic conversion
model = CustomModel(dynamic=True)  # or
model.compile(optimizer="adam", loss="mse", run_eagerly=True)
```

## Key Takeaways

- **95% of use cases** only need tf.keras and tf.data
- **Custom components** enable specialized architectures: losses, metrics, layers, models
- **Variables** are mutable tensors for weights and parameters
- **Autodiff** efficiently computes gradients automatically
- **Custom training loops** provide maximum flexibility but increase complexity
- **TF Functions** boost performance through graph optimization
- **AutoGraph** converts Python control flow to TensorFlow operations
- **Best practice**: Use high-level APIs when possible, customize only when necessary

## Exercise Highlights

1. **Layer Normalization**: Implement custom layer with trainable α and β
2. **Custom training loop**: Train Fashion MNIST with custom display
3. **Different optimizers**: Use separate optimizers for different layer groups

---

*This summary covers the essential concepts from Chapter 12. For complete details, examples, and edge cases, refer to the full chapter.*