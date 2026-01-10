# Chapter 11: Training Deep Neural Networks

## Overview
Training deep neural networks presents unique challenges including vanishing/exploding gradients, insufficient training data, slow training, and overfitting. This chapter covers techniques to address these problems and train very deep networks effectively.

## Key Problems in Training Deep DNNs
- **Vanishing/Exploding Gradients**: Gradients become too small or too large when flowing backward through the network
- **Insufficient Training Data**: Large networks need substantial labeled data
- **Slow Training**: Deep networks can take extremely long to train
- **Overfitting**: Models with millions of parameters easily overfit, especially with limited data

---

## 1. The Vanishing/Exploding Gradients Problems

### Problem Description
During backpropagation, gradients often become progressively smaller (vanishing) or larger (exploding) as they flow backward through layers. This makes lower layers very difficult to train, as Gradient Descent updates leave their weights virtually unchanged or cause divergence.

### Root Causes
Research by Glorot and Bengio (2010) identified key culprits:
- Poor weight initialization (normal distribution with mean=0, std=1)
- Sigmoid activation functions that saturate (output near 0 or 1)
- Variance increases through layers, causing saturation at top layers

When sigmoid functions saturate, their derivatives approach 0, leaving virtually no gradient to propagate backward.

---

## 2. Solutions to Gradient Problems

### Glorot and He Initialization

**Problem**: Random initialization with improper variance causes signal to die out or explode.

**Solution**: Initialize weights so variance of outputs equals variance of inputs.

**Glorot Initialization** (Xavier initialization):

**Equation 11-1:**
- Normal distribution with mean 0 and variance σ² = 1/fan_avg
- Or uniform distribution between -r and +r, with r = √(3/fan_avg)

Where fan_avg = (fan_in + fan_out)/2

**Initialization by Activation Function**:

| Initialization | Activation Functions | σ² (Normal) |
|---------------|---------------------|-------------|
| Glorot | None, tanh, logistic, softmax | 1/fan_avg |
| He | ReLU and variants | 2/fan_in |
| LeCun | SELU | 1/fan_in |

**Keras Implementation**:
```python
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")

# He initialization with uniform distribution based on fan_avg
he_avg_init = keras.initializers.VarianceScaling(
    scale=2., mode='fan_avg', distribution='uniform'
)
keras.layers.Dense(10, activation="sigmoid", kernel_initializer=he_avg_init)
```

### Nonsaturating Activation Functions

**ReLU (Rectified Linear Unit)**: 
- Doesn't saturate for positive values
- Fast to compute
- **Problem**: Dying ReLUs - neurons can "die" and output only 0

**Leaky ReLU**:
- Formula: `LeakyReLU_α(z) = max(αz, z)`
- Typically α=0.01 (small leak)
- Prevents dying ReLUs by allowing small negative slope

**Variants**:
- **Randomized Leaky ReLU (RReLU)**: α picked randomly during training
- **Parametric Leaky ReLU (PReLU)**: α learned via backpropagation

**ELU (Exponential Linear Unit)**:

**Equation 11-2:**
```
ELU_α(z) = α(exp(z) - 1)  if z < 0
         = z               if z ≥ 0
```

Advantages:
- Takes negative values (mean closer to 0)
- Nonzero gradient for z < 0 (avoids dead neurons)
- Smooth everywhere when α=1 (faster Gradient Descent)

**SELU (Scaled ELU)**:
- Scaled variant of ELU for self-normalizing networks
- Network output tends to preserve mean=0 and std=1 during training

**Requirements for SELU self-normalization**:
1. Input features must be standardized (mean=0, std=1)
2. Use LeCun normal initialization: `kernel_initializer="lecun_normal"`
3. Architecture must be sequential (dense layers only)

**Recommendation Hierarchy**:
```
SELU > ELU > leaky ReLU > ReLU > tanh > logistic
```

**Keras Implementation**:
```python
# Leaky ReLU
model = keras.models.Sequential([
    keras.layers.Dense(10, kernel_initializer="he_normal"),
    keras.layers.LeakyReLU(alpha=0.2),
])

# PReLU
keras.layers.PReLU()

# SELU
layer = keras.layers.Dense(10, activation="selu",
                          kernel_initializer="lecun_normal")
```

---

## 3. Batch Normalization

### Concept
Batch Normalization (BN) adds an operation before/after each hidden layer's activation that zero-centers and normalizes inputs, then scales and shifts the result. This significantly reduces vanishing/exploding gradients problems.

### Algorithm

**Equation 11-3: Batch Normalization Algorithm**

Step 1: Compute mean over mini-batch
```
μ_B = (1/m_B) × Σ(i=1 to m_B) x^(i)
```

Step 2: Compute variance over mini-batch
```
σ²_B = (1/m_B) × Σ(i=1 to m_B) (x^(i) - μ_B)²
```

Step 3: Normalize (zero-center and standardize)
```
x̂^(i) = (x^(i) - μ_B) / √(σ²_B + ε)
```

Step 4: Scale and shift
```
z^(i) = γ ⊗ x̂^(i) + β
```

Where:
- **μ_B**: mean vector over mini-batch
- **σ_B**: standard deviation vector over mini-batch
- **m_B**: number of instances in mini-batch
- **γ**: output scale parameter vector (learned)
- **β**: output shift parameter vector (learned)
- **ε**: smoothing term (typically 10⁻⁵)

### Key Parameters
Four parameters per input:
- **γ** (gamma) and **β** (beta): Learned via backpropagation
- **μ** (mu) and **σ** (sigma): Estimated using exponential moving average during training

### Benefits
1. Greatly improves training of deep networks
2. Reduces sensitivity to weight initialization
3. Allows higher learning rates (faster training)
4. Acts as a regularizer (reduces need for dropout)
5. Networks less sensitive to activation function choice

### Drawbacks
- Adds complexity to the model
- Runtime penalty (slower predictions)
- Can be fused with previous layer after training to eliminate overhead

### Keras Implementation

```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])
```

**BN Before Activation** (alternative approach):
```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(10, activation="softmax")
])
```

**Important Hyperparameters**:
- **momentum**: Controls exponential moving average (default 0.99)
  - `v̂ ← v̂ × momentum + v × (1 - momentum)`
  - Use more 9s (0.999) for larger datasets
- **axis**: Which axis to normalize (default -1 for last axis)

### Training vs Testing Behavior
- **Training**: Uses batch statistics (μ_B, σ_B)
- **Testing**: Uses final statistics (moving averages)
- The `call()` method has a `training` argument to control behavior

---

## 4. Gradient Clipping

For recurrent neural networks or when BN isn't sufficient, clip gradients during backpropagation to prevent explosion.

**Clip by Value**:
```python
optimizer = keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss="mse", optimizer=optimizer)
```
Clips each gradient component between -1.0 and 1.0 (may change gradient direction).

**Clip by Norm** (preserves direction):
```python
optimizer = keras.optimizers.SGD(clipnorm=1.0)
```
Clips entire gradient if its ℓ2 norm exceeds threshold.

---

## 5. Reusing Pretrained Layers (Transfer Learning)

### Concept
Instead of training from scratch, reuse lower layers from a model trained on a similar task. This dramatically speeds up training and requires less data.

### Strategy
1. **Find similar task**: Look for existing model solving related problem
2. **Reuse lower layers**: They learn general features (edges, textures)
3. **Replace output layer**: Add new output for your specific task
4. **Freeze & train**: Initially freeze reused layers, train new layers
5. **Fine-tune**: Unfreeze some top layers and train with lower learning rate

### Guidelines
- **More similar tasks** → reuse more layers
- **More training data** → unfreeze more layers
- **Less training data** → keep more layers frozen
- Start with frozen layers, then progressively unfreeze from top down
- Reduce learning rate when unfreezing to avoid destroying fine-tuned weights

### Keras Implementation

```python
# Load pretrained model
model_A = keras.models.load_model("my_model_A.h5")

# Create new model reusing all but output layer
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

# Clone to avoid sharing (optional)
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())

# Freeze reused layers
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd",
                     metrics=["accuracy"])

# Train with frozen layers
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                           validation_data=(X_valid_B, y_valid_B))

# Unfreeze and fine-tune with lower learning rate
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = keras.optimizers.SGD(lr=1e-4)  # reduced learning rate
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
                     metrics=["accuracy"])

history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                           validation_data=(X_valid_B, y_valid_B))
```

**Important**: Always compile after freezing/unfreezing layers!

---

## 6. Unsupervised Pretraining

When you lack labeled data but have abundant unlabeled data:

1. **Train unsupervised model**: Use autoencoder or GAN on unlabeled data
2. **Reuse lower layers**: Transfer learned features to your supervised task
3. **Add output layer**: For your specific task
4. **Fine-tune**: Train with labeled data

This technique revived Deep Learning in 2006 (originally with RBMs). Modern approaches use autoencoders or GANs.

---

## 7. Pretraining on Auxiliary Task

When you have little labeled data for your target task:

**Strategy**: Train on a related task with abundant labeled data, then transfer.

**Example - Face Recognition**:
- **Auxiliary task**: Train to detect if two pictures show same person
- **Lots of data available**: Random faces from web
- **Transfer**: Reuse learned feature detectors for face classification

**Example - NLP**:
- **Auxiliary task**: Predict masked words in sentences
- **Self-supervised**: Automatically generate labels from text corpus
- **Transfer**: Reuse language understanding for downstream tasks

**Self-supervised learning**: Automatically generate labels from data itself, then use supervised learning techniques.

---

## 8. Faster Optimizers

Beyond standard Gradient Descent: θ ← θ - η∇_θJ(θ)

### Momentum Optimization

**Concept**: Like a bowling ball rolling downhill, accumulates velocity.

**Equation 11-4: Momentum Algorithm**

Step 1: Update momentum vector
```
m ← βm - η∇_θJ(θ)
```

Step 2: Update parameters
```
θ ← θ + m
```

- **β** (momentum): Typically 0.9 (friction parameter, range 0-1)
- **Terminal velocity**: gradient × η × 1/(1-β)
- With β=0.9, moves 10× faster than regular Gradient Descent
- Helps escape plateaus and navigate elongated valleys

```python
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)
```

### Nesterov Accelerated Gradient (NAG)

**Improvement**: Measure gradient slightly ahead in momentum direction.

**Equation 11-5: Nesterov Accelerated Gradient Algorithm**

Step 1: Update momentum vector (using lookahead gradient)
```
m ← βm - η∇_θJ(θ + βm)
```

Step 2: Update parameters
```
θ ← θ + m
```

- More accurate gradient measurement
- Reduces oscillations
- Generally faster than regular momentum

```python
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
```

### AdaGrad

**Concept**: Scale down gradient along steepest dimensions (adaptive learning rate).

**Equation 11-6: AdaGrad Algorithm**

Step 1: Accumulate squared gradients
```
s ← s + ∇_θJ(θ) ⊗ ∇_θJ(θ)
```

Step 2: Update parameters (scaled by accumulated gradients)
```
θ ← θ - η∇_θJ(θ) ⊘ √(s + ε)
```

Where:
- ⊗ represents element-wise multiplication
- ⊘ represents element-wise division
- ε is a smoothing term (typically 10⁻¹⁰)

- Accumulates squared gradients in s
- Scales updates by inverse of accumulated gradient
- Points updates toward global optimum
- **Problem**: Can stop too early (learning rate decays too fast)
- Not recommended for deep neural networks

### RMSProp

**Fix for AdaGrad**: Only accumulate recent gradients using exponential decay.

**Equation 11-7: RMSProp Algorithm**

Step 1: Accumulate squared gradients with exponential decay
```
s ← βs + (1-β)∇_θJ(θ) ⊗ ∇_θJ(θ)
```

Step 2: Update parameters
```
θ ← θ - η∇_θJ(θ) ⊘ √(s + ε)
```

Where:
- β (decay rate): Typically 0.9
- ⊗ represents element-wise multiplication
- ⊘ represents element-wise division

```python
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
```

### Adam (Adaptive Moment Estimation)

**Combines**: Momentum + RMSProp

**Equation 11-8**:
```
1. m ← β₁m - (1-β₁)∇_θJ(θ)              # momentum
2. s ← β₂s + (1-β₂)∇_θJ(θ) ⊗ ∇_θJ(θ)    # RMSProp
3. m̂ ← m / (1-β₁ᵗ)                      # bias correction
4. ŝ ← s / (1-β₂ᵗ)                      # bias correction
5. θ ← θ + ηm̂ ⊘ √(ŝ + ε)
```

- **β₁**: Typically 0.9 (momentum decay)
- **β₂**: Typically 0.999 (scaling decay)
- **ε**: Typically 10⁻⁷
- Adaptive learning rate requires less tuning
- **Default η=0.001** often works well

```python
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
```

**Variants**:
- **AdaMax**: Uses ℓ∞ norm instead of ℓ2 (more stable in some cases)
- **Nadam**: Adam + Nesterov trick (often converges slightly faster)

**Note**: Adaptive methods may generalize poorly on some datasets. Try Nesterov SGD if disappointed with Adam.

### Optimizer Comparison

| Optimizer | Convergence Speed | Convergence Quality |
|-----------|------------------|-------------------|
| SGD | ★ | ★★★ |
| SGD (momentum) | ★★ | ★★★ |
| SGD (Nesterov) | ★★ | ★★★ |
| Adagrad | ★★★ | ★ (stops early) |
| RMSprop | ★★★ | ★★ or ★★★ |
| Adam | ★★★ | ★★ or ★★★ |
| Nadam | ★★★ | ★★ or ★★★ |
| AdaMax | ★★★ | ★★ or ★★★ |

---

## 9. Learning Rate Scheduling

Find good learning rate faster by varying it during training.

### Common Schedules

**Power Scheduling**:
```
η(t) = η₀ / (1 + t/s)^c
```
- Drops quickly then slowly
- Requires tuning η₀, s, and c

**Exponential Scheduling**:
```
η(t) = η₀ × 0.1^(t/s)
```
- Drops by factor of 10 every s steps

**Piecewise Constant**:
- Different constant rates for different epoch ranges
- Example: η=0.1 for 5 epochs, η=0.005 for 10 epochs, η=0.001 after

**Performance Scheduling**:
- Reduce learning rate when validation error plateaus
- Example: Multiply by 0.5 after 5 epochs without improvement

**1cycle Scheduling**:
1. Increase η₀ to η₁ (first half of training)
2. Decrease η₁ back to η₀ (second half)
3. Drop by orders of magnitude (final epochs)
- Also varies momentum inversely
- Often achieves better performance faster

### Keras Implementation

**Power Scheduling** (simplest):
```python
optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)
```

**Exponential Scheduling**:
```python
def exponential_decay_fn(epoch):
    return 0.01 * 0.1**(epoch / 20)

lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train, y_train, callbacks=[lr_scheduler])
```

**Piecewise Constant**:
```python
def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001

lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)
```

**Performance Scheduling**:
```python
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
```

**Using schedules API** (updates at each step):
```python
s = 20 * len(X_train) // 32  # steps in 20 epochs
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimizer = keras.optimizers.SGD(learning_rate)
```

---

## 10. Regularization Techniques

### ℓ1 and ℓ2 Regularization

Constrain connection weights to prevent overfitting.

```python
# ℓ2 regularization
layer = keras.layers.Dense(100, activation="elu",
                          kernel_initializer="he_normal",
                          kernel_regularizer=keras.regularizers.l2(0.01))

# ℓ1 regularization (for sparse models)
kernel_regularizer=keras.regularizers.l1(0.01)

# Both ℓ1 and ℓ2
kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)
```

**Avoid repetition with partial()**:
```python
from functools import partial

RegularizedDense = partial(keras.layers.Dense,
                          activation="elu",
                          kernel_initializer="he_normal",
                          kernel_regularizer=keras.regularizers.l2(0.01))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    RegularizedDense(300),
    RegularizedDense(100),
    RegularizedDense(10, activation="softmax")
])
```

### Dropout

**Concept**: Randomly "drop" neurons during training (set output to 0).

**How it works**:
- Each training step: every neuron has probability p of being dropped
- **p** (dropout rate): Typically 0.2-0.5
- Prevents co-adaptation of neurons
- Forces network to be robust
- Like training an ensemble of 2^N networks

**Technical detail**: Divide remaining outputs by keep probability (1-p) during training to compensate.

**Keras Implementation**:
```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
])
```

**Guidelines**:
- Increase dropout rate if overfitting
- Decrease if underfitting
- Higher rates for large layers, lower for small layers
- Many architectures only use dropout after last hidden layer
- Slows convergence but improves final model

**Alpha Dropout**: For self-normalizing networks (SELU), preserves mean and std.

### Monte Carlo (MC) Dropout

**Powerful technique** from 2016 paper by Gal and Ghahramani:

**Benefits**:
1. Boosts any trained dropout model (no retraining needed)
2. Provides better uncertainty estimates
3. Simple to implement

**Implementation**:
```python
# Make 100 predictions with dropout active
y_probas = np.stack([model(X_test_scaled, training=True)
                    for sample in range(100)])
y_proba = y_probas.mean(axis=0)

# Get uncertainty estimates
y_std = y_probas.std(axis=0)
```

**Results**:
- More reliable predictions than single prediction
- Uncertainty estimates (standard deviation)
- Typically small accuracy boost
- Trade-off: More samples = better estimates but slower inference

**Custom MCDropout class** (for models with BatchNorm):
```python
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
```

### Max-Norm Regularization

**Concept**: Constrain incoming connection weights per neuron.

**Constraint**: ‖w‖₂ ≤ r

- After each training step, rescale w if needed: w ← w × r/‖w‖₂
- Reduces overfitting
- Helps with unstable gradients

```python
keras.layers.Dense(100, activation="elu", 
                  kernel_initializer="he_normal",
                  kernel_constraint=keras.constraints.max_norm(1.))
```

Can also constrain biases with `bias_constraint`.

---

## 11. Training Sparse Models

For faster runtime or reduced memory:

**Approaches**:
1. **Post-training pruning**: Train normally, then zero out tiny weights
2. **ℓ1 regularization**: Encourages sparsity during training
3. **TensorFlow Model Optimization Toolkit (TF-MOT)**: Iteratively removes connections by magnitude during training

---

## Summary and Practical Guidelines

### Default DNN Configuration

| Hyperparameter | Default Value |
|---------------|---------------|
| Kernel initializer | He initialization |
| Activation function | ELU |
| Normalization | None (shallow); Batch Norm (deep) |
| Regularization | Early stopping (+ ℓ2 if needed) |
| Optimizer | Momentum (or RMSProp/Nadam) |
| Learning rate schedule | 1cycle |

### Self-Normalizing Net Configuration

| Hyperparameter | Default Value |
|---------------|---------------|
| Kernel initializer | LeCun initialization |
| Activation function | SELU |
| Normalization | None (self-normalizing) |
| Regularization | Alpha dropout if needed |
| Optimizer | Momentum (or RMSProp/Nadam) |
| Learning rate schedule | 1cycle |

### Additional Guidelines

**Always**:
- Normalize input features
- Try to reuse pretrained networks when possible
- Use unsupervised pretraining if abundant unlabeled data
- Use auxiliary task pretraining if labeled data for similar task

**Exceptions**:
- **Sparse models**: Use ℓ1 regularization or TF-MOT (breaks self-normalization)
- **Low-latency models**: Fewer layers, fold BN layers, leaky ReLU, sparse model, reduce precision (16/8-bit)
- **Risk-sensitive applications**: Use MC Dropout for better probability estimates and uncertainty

### Key Takeaways

1. **Gradient problems**: Use proper initialization (He/Glorot) and activation functions (ELU/SELU/ReLU)
2. **Batch Normalization**: Powerful technique for deep networks, acts as regularizer
3. **Transfer Learning**: Reuse pretrained models whenever possible
4. **Optimizers**: Adam/Nadam work well generally; try Nesterov SGD if adaptive methods underperform
5. **Learning rate scheduling**: 1cycle or exponential decay speeds up training
6. **Regularization**: Early stopping + Dropout + ℓ2 prevent overfitting
7. **MC Dropout**: Boosts models and provides uncertainty estimates

---

## Exercises

1. Is it OK to initialize all weights to the same value with He initialization?
2. Is it OK to initialize bias terms to 0?
3. Name three advantages of SELU over ReLU.
4. When would you use: SELU, leaky ReLU, ReLU, tanh, logistic, softmax?
5. What happens if momentum is too close to 1 (e.g., 0.99999)?
6. Name three ways to produce a sparse model.
7. Does dropout slow training? Does it slow inference? What about MC Dropout?
8. Practice: Train deep neural network on CIFAR10 with:
   - 20 hidden layers of 100 neurons each
   - He initialization and ELU
   - Nadam optimization and early stopping
   - Then try Batch Normalization
   - Then try SELU with self-normalization
   - Then try alpha dropout and MC Dropout
   - Finally try 1cycle scheduling

---

**With these techniques, you're ready to train very deep neural networks effectively!**