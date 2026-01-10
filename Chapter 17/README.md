# Chapter 17: Representation Learning and Generative Learning Using Autoencoders and GANs

## Overview

This chapter covers two major unsupervised learning approaches for representation and generative learning:

- **Autoencoders**: Neural networks that learn compressed representations (codings) of input data without supervision
- **GANs (Generative Adversarial Networks)**: Networks with competing components that generate highly realistic synthetic data

Both can be used for dimensionality reduction, feature extraction, and generating new data, but they work very differently.

## Part 1: Autoencoders

### What Are Autoencoders?

Autoencoders learn to copy their inputs to outputs while being constrained in various ways. This forces them to learn efficient data representations rather than trivially copying.

**Key Components:**
- **Encoder (Recognition Network)**: Converts inputs to latent representations
- **Decoder (Generative Network)**: Reconstructs outputs from latent representations

**Applications:**
- Dimensionality reduction and visualization
- Feature extraction for supervised learning
- Unsupervised pretraining of deep networks
- Denoising and image generation

### PCA with Linear Autoencoders

A simple linear autoencoder with MSE loss performs Principal Component Analysis:

```python
from tensorflow import keras

encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[3])])
decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape=[2])])
autoencoder = keras.models.Sequential([encoder, decoder])

autoencoder.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.1))
history = autoencoder.fit(X_train, X_train, epochs=20)
codings = encoder.predict(X_train)
```

Note: Training uses `X_train` as both inputs and targets (self-supervised learning).

### Stacked Autoencoders

Deep autoencoders with multiple hidden layers learn more complex representations.

**Architecture**: Typically symmetrical around the central coding layer (e.g., 784 → 100 → 30 → 100 → 784).

```python
stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),
])

stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
stacked_ae.compile(loss="binary_crossentropy",
                   optimizer=keras.optimizers.SGD(lr=1.5))
history = stacked_ae.fit(X_train, X_train, epochs=10,
                        validation_data=[X_valid, X_valid])
```

**Why binary cross-entropy?** Treating reconstruction as multilabel binary classification (each pixel = probability of being black) often converges faster than MSE.

### Applications of Stacked Autoencoders

#### 1. Dimensionality Reduction for Visualization

Combine autoencoders with t-SNE for effective visualization:

```python
from sklearn.manifold import TSNE

X_valid_compressed = stacked_encoder.predict(X_valid)
tsne = TSNE()
X_valid_2D = tsne.fit_transform(X_valid_compressed)
plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap="tab10")
```

#### 2. Unsupervised Pretraining

When you have lots of unlabeled data but few labeled examples:

1. Train autoencoder on all data (labeled + unlabeled)
2. Reuse encoder layers in supervised model
3. Train classifier on labeled data (optionally freeze pretrained layers)

This is valuable because unlabeled data is cheap while labeling is expensive.

### Advanced Training Techniques

#### Tying Weights

For symmetrical autoencoders, tie decoder weights to encoder weights (transposed): `W_{N-L+1} = W_L^T`

```python
class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)
    
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias", initializer="zeros",
                                     shape=[self.dense.input_shape[-1]])
        super().build(batch_input_shape)
    
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)

# Usage
dense_1 = keras.layers.Dense(100, activation="selu")
dense_2 = keras.layers.Dense(30, activation="selu")

tied_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    dense_1,
    dense_2
])

tied_decoder = keras.models.Sequential([
    DenseTranspose(dense_2, activation="selu"),
    DenseTranspose(dense_1, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
```

**Benefits**: Halves the number of parameters, speeds up training, reduces overfitting.

#### Greedy Layer-wise Training

Historical approach (less common now): Train one shallow autoencoder at a time, then stack them.

### Specialized Autoencoder Architectures

#### Convolutional Autoencoders

For image data, use CNNs instead of dense layers:

```python
conv_encoder = keras.models.Sequential([
    keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="selu"),
    keras.layers.MaxPool2D(pool_size=2)
])

conv_decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, 
                                 padding="valid", activation="selu",
                                 input_shape=[3, 3, 64]),
    keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, 
                                 padding="same", activation="selu"),
    keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, 
                                 padding="same", activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
```

**Key**: Encoder downsamples (reduces spatial dimensions, increases depth), decoder upsamples (opposite).

#### Recurrent Autoencoders

For sequences (time series, text):

```python
recurrent_encoder = keras.models.Sequential([
    keras.layers.LSTM(100, return_sequences=True, input_shape=[None, 28]),
    keras.layers.LSTM(30)
])

recurrent_decoder = keras.models.Sequential([
    keras.layers.RepeatVector(28, input_shape=[30]),
    keras.layers.LSTM(100, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(28, activation="sigmoid"))
])
```

**Architecture**: Sequence-to-vector encoder compresses input, vector-to-sequence decoder reconstructs.

### Denoising Autoencoders

Add noise to inputs, train to recover original noise-free inputs. Forces learning robust features.

```python
dropout_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(0.5),  # Randomly drop 50% of pixels
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu")
])

dropout_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
```

**Noise types**: Gaussian noise or dropout (randomly switched-off inputs).

**Applications**: Data visualization, unsupervised pretraining, and noise removal.

### Sparse Autoencoders

Force the coding layer to have few active neurons, encouraging meaningful feature extraction.

#### Approach 1: L1 Regularization

```python
sparse_l1_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(300, activation="sigmoid"),  # Large coding layer
    keras.layers.ActivityRegularization(l1=1e-3)
])
```

#### Approach 2: KL Divergence (Better Results)

Penalize when actual sparsity differs from target sparsity.

**KL Divergence Formula:**
```
D_KL(p || q) = p log(p/q) + (1-p) log((1-p)/(1-q))
```

Where `p` = target sparsity, `q` = actual mean activation.

```python
K = keras.backend
kl_divergence = keras.losses.kullback_leibler_divergence

class KLDivergenceRegularizer(keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1):
        self.weight = weight
        self.target = target
    
    def __call__(self, inputs):
        mean_activities = K.mean(inputs, axis=0)
        return self.weight * (
            kl_divergence(self.target, mean_activities) +
            kl_divergence(1. - self.target, 1. - mean_activities))

kld_reg = KLDivergenceRegularizer(weight=0.05, target=0.1)
sparse_kl_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(300, activation="sigmoid", activity_regularizer=kld_reg)
])
```

**Result**: ~70% of activations < 0.1, mean activation per neuron ≈ 0.1.

### Variational Autoencoders (VAEs)

**Key Differences from Standard Autoencoders:**
- **Probabilistic**: Outputs partly determined by chance
- **Generative**: Can generate new instances resembling training data

**How They Work:**

1. Encoder outputs mean (μ) and standard deviation (σ)
2. Sample coding from Gaussian distribution: `z ~ N(μ, σ²)`
3. Decoder reconstructs from sampled coding

```python
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean
```

**Architecture:**

```python
codings_size = 10

# Encoder
inputs = keras.layers.Input(shape=[28, 28])
z = keras.layers.Flatten()(inputs)
z = keras.layers.Dense(150, activation="selu")(z)
z = keras.layers.Dense(100, activation="selu")(z)
codings_mean = keras.layers.Dense(codings_size)(z)  # μ
codings_log_var = keras.layers.Dense(codings_size)(z)  # γ = log(σ²)
codings = Sampling()([codings_mean, codings_log_var])

variational_encoder = keras.Model(
    inputs=[inputs], 
    outputs=[codings_mean, codings_log_var, codings])

# Decoder
decoder_inputs = keras.layers.Input(shape=[codings_size])
x = keras.layers.Dense(100, activation="selu")(decoder_inputs)
x = keras.layers.Dense(150, activation="selu")(x)
x = keras.layers.Dense(28 * 28, activation="sigmoid")(x)
outputs = keras.layers.Reshape([28, 28])(x)

variational_decoder = keras.Model(inputs=[decoder_inputs], outputs=[outputs])

# Complete VAE
_, _, codings = variational_encoder(inputs)
reconstructions = variational_decoder(codings)
variational_ae = keras.Model(inputs=[inputs], outputs=[reconstructions])
```

**Loss Function:**

Two components:
1. **Reconstruction loss**: Binary cross-entropy (as before)
2. **Latent loss**: KL divergence pushing codings toward Gaussian distribution

**Latent Loss Formula (using γ = log(σ²)):**
```
L = -1/2 * Σ[1 + γ_i - exp(γ_i) - μ_i²]
```

```python
latent_loss = -0.5 * K.sum(
    1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
    axis=-1)

variational_ae.add_loss(K.mean(latent_loss) / 784.)
variational_ae.compile(loss="binary_crossentropy", optimizer="rmsprop")

history = variational_ae.fit(X_train, X_train, epochs=50, batch_size=128,
                            validation_data=[X_valid, X_valid])
```

**Generating New Images:**

```python
codings = tf.random.normal(shape=[12, codings_size])
images = variational_decoder(codings).numpy()
```

**Semantic Interpolation:**

Interpolate in latent space (not pixel space) to get meaningful transitions:

```python
codings_grid = tf.reshape(codings, [1, 3, 4, codings_size])
larger_grid = tf.image.resize(codings_grid, size=[5, 7])  # Bilinear interpolation
interpolated_codings = tf.reshape(larger_grid, [-1, codings_size])
images = variational_decoder(interpolated_codings).numpy()
```

## Part 2: Generative Adversarial Networks (GANs)

### What Are GANs?

GANs use competition between two networks to generate highly realistic data.

**Components:**

1. **Generator**: Creates fake data from random noise
2. **Discriminator**: Distinguishes real data from fake

**Analogy**: Generator = counterfeiter making fake money, Discriminator = detective identifying fakes.

**Training Process** (two phases per iteration):

**Phase 1 - Train Discriminator:**
- Generate fake images from random noise
- Combine with equal number of real images
- Labels: 0 for fake, 1 for real
- Train discriminator (generator frozen)

**Phase 2 - Train Generator:**
- Generate fake images
- Labels: all 1 (want discriminator to think they're real)
- Train generator (discriminator frozen)

### Basic GAN Implementation

```python
codings_size = 30

# Generator
generator = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[codings_size]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

# Discriminator
discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(1, activation="sigmoid")
])

# Complete GAN
gan = keras.models.Sequential([generator, discriminator])

# Compile
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False  # Only for GAN model
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")
```

**Training Loop:**

```python
def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        for X_batch in dataset:
            # Phase 1 - Train discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            
            # Phase 2 - Train generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)

# Prepare dataset
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

train_gan(gan, dataset, batch_size, codings_size)
```

### Challenges in Training GANs

#### 1. Mode Collapse

Generator produces limited variety (e.g., only shoes, then only shirts, cycling through classes).

**Solution techniques:**
- Experience replay: Store generated images in buffer, train discriminator on mix
- Mini-batch discrimination: Measure similarity across batch, help discriminator reject non-diverse outputs

#### 2. Training Instability

Parameters can oscillate, training may suddenly diverge.

**Contributing factors:**
- Competing networks pushing against each other
- Sensitive to hyperparameters
- May never reach Nash equilibrium (perfect generator, guessing discriminator)

#### 3. Evaluation Difficulty

Hard to automatically judge image quality (diversity easier to measure).

### Deep Convolutional GANs (DCGANs)

**Key Guidelines** (Radford et al., 2015):

1. Replace pooling with strided convolutions (discriminator) and transposed convolutions (generator)
2. Use Batch Normalization (except first/last layers)
3. Remove fully connected hidden layers for deeper architectures
4. Generator: ReLU (hidden), tanh (output)
5. Discriminator: Leaky ReLU (all layers)

**Implementation:**

```python
codings_size = 100

generator = keras.models.Sequential([
    keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
    keras.layers.Reshape([7, 7, 128]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, 
                                 padding="same", activation="selu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, 
                                 padding="same", activation="tanh")
])

discriminator = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same",
                       activation=keras.layers.LeakyReLU(0.2),
                       input_shape=[28, 28, 1]),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same",
                       activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")
])

# Rescale training data to [-1, 1] for tanh output
X_train = X_train.reshape(-1, 28, 28, 1) * 2. - 1.
```

**Capabilities:**
- Generate fairly realistic images
- Learn meaningful latent representations
- Enable arithmetic on faces (e.g., "man with glasses" - "man without glasses" + "woman without glasses" = "woman with glasses")

**Conditional GANs (CGANs)**: Add class labels as input to both networks to control generated class.

### Progressive Growing of GANs

**Key Idea**: Start with small images (4×4), gradually add layers to generate larger images (8×8, 16×16, ..., 1024×1024).

**Fade-in Technique:**
- New layers gradually faded in (weight α: 0→1)
- Old output layers faded out (weight 1-α: 1→0)
- Prevents breaking trained weights

**Additional Techniques:**

1. **Minibatch Standard Deviation Layer**
   - Computes standard deviation across batch
   - Helps discriminator detect lack of diversity
   - Reduces mode collapse risk

2. **Equalized Learning Rate**
   - Initialize weights: N(0, 1)
   - Scale at runtime: divide by √(2/n_inputs)
   - Ensures all parameters learn at same speed

3. **Pixelwise Normalization**
   - Added after each generator conv layer
   - Normalizes across channels at each location
   - Prevents activation explosions

**Result**: Extremely convincing high-resolution face generation.

### StyleGANs

State-of-the-art architecture (Karras et al., 2018) using style transfer for high-quality images.

**Components:**

1. **Mapping Network**
   - 8-layer MLP
   - Maps latent codes z → style vectors w
   - Multiple affine transformations produce style vectors for different levels

2. **Synthesis Network**
   - Learned constant input (not random noise)
   - Convolutions + upsampling layers
   - **Noise injection**: Added at each layer for stochastic details (freckles, hair position)
   - **AdaIN (Adaptive Instance Normalization)**: Standardizes feature maps, then scales/shifts using style vectors

**Key Innovations:**

1. **Separate Noise Inputs**
   - Avoids wasting coding capacity on randomness
   - Different noise per level for proper stochasticity
   - Each noise: single feature map broadcasted and scaled

2. **Style Mixing (Mixing Regularization)**
   - Generate images using two codings
   - Use styles w1 for early levels, w2 for later levels
   - Encourages locality: each style affects limited traits

**Architecture Flow:**
```
Random z → Mapping Net → Style vectors w → Synthesis Net (with noise) → Image
```

## Key Takeaways

### Autoencoders
- Learn compressed representations through self-supervision
- Undercomplete (smaller coding) forces meaningful features
- Variants: stacked, convolutional, recurrent, denoising, sparse, variational
- Applications: dimensionality reduction, pretraining, denoising, generation

### GANs
- Two competing networks: generator creates, discriminator judges
- Training is challenging (mode collapse, instability)
- DCGANs: architectural guidelines for stable training
- Progressive GANs: grow from small to large images
- StyleGANs: state-of-the-art quality via style transfer

### Comparison
- **Autoencoders**: Learn by reconstruction with constraints
- **GANs**: Learn through adversarial competition
- **VAEs vs GANs**: VAEs produce fuzzier images but are easier to train; GANs produce sharper, more realistic images

## Practical Tips

1. **Autoencoders**: Start simple, add complexity as needed
2. **GANs**: Expect training difficulties, be patient
3. **Hyperparameters**: Both are sensitive; extensive tuning often required
4. **Pretrained Models**: Use when possible for quick results
5. **Evaluation**: Use human raters for quality; diversity can be automated

## Further Exploration

- Experiment with different architectures
- Try conditional generation (CGANs)
- Explore advanced GAN variants (ProGAN, BigGAN, etc.)
- Use pretrained models for transfer learning
- Study mathematical foundations for deeper understanding