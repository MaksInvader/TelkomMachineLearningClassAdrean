# Deep Computer Vision Using Convolutional Neural Networks

## Introduction

While computers could beat chess champions in 1996, recognizing simple objects like puppies in images remained challenging until recently. This difficulty exists because human perception occurs unconsciously in specialized brain modules. CNNs emerged from studying the brain's visual cortex and have achieved superhuman performance on complex visual tasks, powering image search, self-driving cars, and video classification systems.

## The Architecture of the Visual Cortex

Hubel and Wiesel's experiments (1958-1959) on cats revealed key insights about visual cortex structure:

- Many neurons have **small local receptive fields** - they respond only to stimuli in limited visual regions
- Receptive fields of different neurons overlap and tile the entire visual field
- Some neurons react only to specific line orientations (horizontal, vertical, etc.)
- Higher-level neurons have larger receptive fields and detect complex patterns by combining lower-level patterns
- This hierarchical architecture enables detection of complex patterns across the visual field

This inspired the **neocognitron** (1980), which evolved into modern CNNs. The landmark **LeNet-5** architecture (1998) by Yann LeCun introduced two crucial building blocks: **convolutional layers** and **pooling layers**.

**Why not use fully connected DNNs?** A 100×100 pixel image with a first layer of 1,000 neurons requires 10 million connections - CNNs solve this using partially connected layers and weight sharing.

## Convolutional Layers

### Basic Architecture

In convolutional layers, neurons connect only to pixels in their **receptive fields**, not to every pixel. This architecture:
- Concentrates on small low-level features in early layers
- Assembles them into larger high-level features in subsequent layers
- Matches the hierarchical structure of real-world images

Each layer is represented in 2D (unlike flattened layers in regular DNNs), making it easier to match neurons with their inputs.

### Key Concepts

**Receptive Fields:** A neuron at row i, column j connects to neurons in the previous layer at rows i to i + f_h - 1, columns j to j + f_w - 1, where f_h and f_w are the receptive field dimensions.

**Zero Padding:** Adding zeros around inputs to maintain layer dimensions (labeled as "same" padding).

**Stride:** The shift between receptive fields. A stride of 2 dramatically reduces computational complexity. A neuron at position (i, j) with stride (s_h, s_w) connects to inputs at rows i × s_h to i × s_h + f_h - 1.

### Filters (Convolution Kernels)

Filters are small weight matrices the size of receptive fields. Examples:
- A vertical line filter (7×7 matrix with 1s in the central column, 0s elsewhere) enhances vertical lines
- A horizontal line filter enhances horizontal lines

When all neurons in a layer use the same filter, they produce a **feature map** highlighting areas that activate the filter most. During training, convolutional layers automatically learn the most useful filters.

### Stacking Multiple Feature Maps

A convolutional layer actually outputs multiple feature maps (in 3D):
- One feature map per filter
- One neuron per pixel in each feature map
- All neurons within a feature map share the same parameters (dramatically reducing model parameters)
- Neurons in different feature maps use different parameters

**Key advantage:** Once the CNN learns to recognize a pattern in one location, it can recognize it anywhere (translation invariance). Regular DNNs must relearn patterns for each location.

### Mathematical Formulation

The output of a neuron in a convolutional layer:

```
z_i,j,k = b_k + Σ(u=0 to f_h-1) Σ(v=0 to f_w-1) Σ(k'=0 to f_n'-1) x_i',j',k' · w_u,v,k',k

where:
i' = i × s_h + u
j' = j × s_w + v
```

Where:
- z_i,j,k: output of neuron at row i, column j in feature map k
- s_h, s_w: vertical and horizontal strides
- f_h, f_w: receptive field height and width
- f_n': number of feature maps in previous layer
- x_i',j',k': input from previous layer at position (i', j', k')
- b_k: bias term for feature map k
- w_u,v,k',k: connection weight

### TensorFlow Implementation

```python
from sklearn.datasets import load_sample_image

# Load and normalize images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])

# Create filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line
filters[3, :, :, 1] = 1  # horizontal line

# Apply convolution
outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")
```

**Padding options:**
- `"SAME"`: Uses zero padding; output size = input size / stride (rounded up)
- `"VALID"`: No padding; may ignore some rows/columns

**Using Keras:**

```python
conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                          padding="same", activation="relu")
```

### Memory Requirements

CNNs require significant RAM, especially during training:

Example: A layer with 5×5 filters outputting 200 feature maps of size 150×100:
- Parameters: (5 × 5 × 3 + 1) × 200 = 15,200
- Computations: 225 million float multiplications
- Memory (32-bit floats): 12 MB per instance, 1.2 GB for batch of 100

**Solutions for out-of-memory errors:**
- Reduce mini-batch size
- Increase stride to reduce dimensionality
- Remove layers
- Use 16-bit floats instead of 32-bit
- Distribute across multiple devices

## Pooling Layers

### Purpose

Pooling layers subsample inputs to:
- Reduce computational load
- Reduce memory usage
- Reduce parameters (limiting overfitting risk)

### How They Work

Neurons connect to limited receptive fields (like convolutional layers) but have **no weights**. They aggregate inputs using functions like max or mean.

**Max Pooling Example:**
- 2×2 pooling kernel, stride 2, no padding
- Only the maximum value in each receptive field propagates forward
- Output is half the height and half the width of input

Pooling layers work independently on each input channel, so output depth equals input depth.

### Benefits

**Translation Invariance:** Max pooling provides some invariance to small translations. Shifting an image by 1-2 pixels may produce identical or nearly identical outputs.

**Other invariance types:** Limited rotational and scale invariance.

### Downsides

- **Destructive:** A 2×2 kernel with stride 2 drops 75% of input values
- **Not always desirable:** For tasks like semantic segmentation, we need **equivariance** (small input changes → corresponding output changes), not invariance

### TensorFlow Implementation

```python
# Max pooling
max_pool = keras.layers.MaxPool2D(pool_size=2)

# Average pooling
avg_pool = keras.layers.AvgPool2D(pool_size=2)
```

Max pooling is generally preferred as it preserves stronger features and provides better translation invariance.

### Depthwise Max Pooling

Pooling can occur along the depth dimension instead of spatial dimensions, helping CNNs learn invariance to features like rotation or thickness.

```python
# Using TensorFlow's low-level API
output = tf.nn.max_pool(images,
                        ksize=(1, 1, 1, 3),
                        strides=(1, 1, 1, 3),
                        padding="valid")

# Wrap in Lambda layer for Keras
depth_pool = keras.layers.Lambda(
    lambda X: tf.nn.max_pool(X, ksize=(1, 1, 1, 3), 
                             strides=(1, 1, 1, 3),
                             padding="valid"))
```

### Global Average Pooling

Computes the mean of each entire feature map, outputting one number per feature map per instance. Extremely destructive but useful as an output layer.

```python
global_avg_pool = keras.layers.GlobalAvgPool2D()

# Equivalent to:
global_avg_pool = keras.layers.Lambda(lambda X: tf.reduce_mean(X, axis=[1, 2]))
```

## CNN Architectures

### Typical Structure

Stack of:
1. Few convolutional layers (+ReLU)
2. Pooling layer
3. Repeat steps 1-2
4. Regular feedforward network (fully connected layers + ReLU)
5. Final output layer (e.g., softmax)

**Best practice:** Use smaller kernels (3×3) stacked rather than large kernels (5×5), except for the first layer which can use larger kernels with stride ≥2.

### Example: Fashion MNIST CNN

```python
model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation="relu", padding="same",
                       input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])
```

Key points:
- Number of filters grows as we go deeper (64 → 128 → 256)
- Common to double filters after each pooling layer
- Flatten before dense layers
- Dropout for regularization

### LeNet-5 (1998)

Created by Yann LeCun for handwritten digit recognition (MNIST).

Architecture:
- Input: 32×32 (MNIST images zero-padded from 28×28)
- C1: Convolution, 6 maps, 28×28, 5×5 kernel, stride 1, tanh
- S2: Avg pooling, 6 maps, 14×14, 2×2 kernel, stride 2, tanh
- C3: Convolution, 16 maps, 10×10, 5×5 kernel, stride 1, tanh
- S4: Avg pooling, 16 maps, 5×5, 2×2 kernel, stride 2, tanh
- C5: Convolution, 120 maps, 1×1, 5×5 kernel, stride 1, tanh
- F6: Fully connected, 84 units, tanh
- Output: Fully connected, 10 units, RBF

### AlexNet (2012)

Won ILSVRC 2012 with 17% top-5 error (vs. 26% for second place).

Key features:
- Much larger and deeper than LeNet-5
- First to stack convolutional layers directly without pooling between them
- Used ReLU activation
- Used dropout (50% rate) on layers F9 and F10
- **Data augmentation:** Random shifts, horizontal flips, lighting changes
- **Local Response Normalization (LRN):** Strongly activated neurons inhibit others in neighboring feature maps

LRN formula:

```
b_i = a_i / (k + α Σ(j=j_low to j_high) a_j²)^β

where:
j_high = min(i + r/2, f_n - 1)
j_low = max(0, i - r/2)
```

Parameters: r=2, α=0.00002, β=0.75, k=1

### Data Augmentation

Artificially increases training set size by generating realistic variants:
- Shift, rotate, resize images
- Adjust contrast and lighting
- Horizontal flips (except for asymmetric objects)
- Combine transformations

Goal: Force model to be tolerant to variations while maintaining realism.

### GoogLeNet (2014)

Won ILSVRC 2014 with <7% top-5 error. Much deeper than predecessors but with 10× fewer parameters than AlexNet (~6M vs 60M).

**Innovation: Inception Modules**

An inception module processes input through four parallel paths:
1. 1×1 convolution
2. 1×1 convolution → 3×3 convolution
3. 1×1 convolution → 5×5 convolution
4. 3×3 max pooling → 1×1 convolution

All paths use stride 1 and "same" padding, so outputs have identical dimensions and can be concatenated along the depth dimension.

**Why 1×1 convolutions?**
1. Capture patterns along depth dimension
2. Act as **bottleneck layers** to reduce dimensionality
3. Pairs like [1×1, 3×3] act as two-layer neural networks, capturing more complex patterns

Architecture:
- Starts with 2 convolutional layers (reducing spatial dimensions by 4)
- LRN layer
- 2 more convolutional layers
- LRN layer
- Max pooling
- Stack of 9 inception modules (with max pooling layers interspersed)
- Global average pooling (eliminates need for multiple dense layers)
- Dropout
- Dense output layer (1000 units, softmax)

### VGGNet (2014)

Runner-up in ILSVRC 2014. Very simple architecture:
- 2-3 convolutional layers → pooling layer (repeated)
- 16 or 19 convolutional layers total
- Only 3×3 filters, but many of them
- Classical, straightforward design

### ResNet (2015)

Won ILSVRC 2015 with <3.6% top-5 error. Extremely deep (152 layers in winning variant, also 34, 50, 101 layer versions).

**Key Innovation: Skip Connections (Shortcut Connections)**

Instead of learning h(x), the network learns f(x) = h(x) - x (residual learning).

Benefits:
1. When initialized, network outputs ~0, so with skip connections it models identity function initially
2. Speeds up training if target function is close to identity
3. Signal easily propagates through entire network
4. Network can make progress even if some layers haven't started learning

Architecture (ResNet-34):
- Initial conv layer (7×7, stride 2)
- Max pooling (3×3, stride 2)
- 3 residual units (64 feature maps)
- 4 residual units (128 feature maps)
- 6 residual units (256 feature maps)
- 3 residual units (512 feature maps)
- Global average pooling
- Dense output layer

Each residual unit:
- 2 convolutional layers (3×3 kernels, stride 1, "same" padding)
- Batch Normalization + ReLU
- Skip connection adding input to output

When feature maps double and spatial dimensions halve, use 1×1 convolution with stride 2 on skip connection to match dimensions.

Deeper ResNets (e.g., ResNet-152) use bottleneck residual units:
- 1×1 conv (reduce channels by 4)
- 3×3 conv
- 1×1 conv (restore original channel count)

### Xception (2016)

Created by François Chollet (Keras author). Merges GoogLeNet and ResNet ideas.

**Key Innovation: Depthwise Separable Convolutions**

Assumes spatial patterns and cross-channel patterns can be modeled separately:
1. **Depthwise convolution:** One spatial filter per input channel
2. **Pointwise convolution:** 1×1 convolution for cross-channel patterns

Benefits:
- Fewer parameters
- Less memory
- Fewer computations
- Often better performance

Architecture:
- 2 regular convolutional layers (start)
- 34 separable convolutional layers
- Few max pooling layers
- Global average pooling + dense output

**Note:** Avoid using separable convolutions after layers with few channels.

### SENet (2017)

Won ILSVRC 2017 with 2.25% top-5 error.

**Innovation: SE Blocks (Squeeze-and-Excitation)**

An SE block analyzes layer output focusing on depth dimension, learning which features are typically active together. It recalibrates feature maps accordingly.

SE Block architecture:
1. **Global average pooling:** Computes mean activation per feature map
2. **Hidden dense layer (ReLU):** Typically 16× fewer neurons than feature maps (squeeze/bottleneck)
3. **Output dense layer (sigmoid):** Outputs recalibration vector (one value per feature map, 0-1 range)
4. Feature maps multiplied by recalibration vector

Can extend existing architectures (SE-Inception, SE-ResNet) by adding SE blocks to each unit.

## Implementing ResNet-34 in Keras

```python
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides,
                              padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1,
                              padding="same", use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides,
                                  padding="same", use_bias=False),
                keras.layers.BatchNormalization()]
    
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

# Build ResNet-34
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224, 3],
                              padding="same", use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"))

prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))
```

## Using Pretrained Models

Keras provides pretrained models in `keras.applications`:

```python
# Load ResNet-50 pretrained on ImageNet
model = keras.applications.resnet50.ResNet50(weights="imagenet")

# Resize images to expected size (224×224 for ResNet-50)
images_resized = tf.image.resize(images, [224, 224])

# Preprocess (each model has specific requirements)
inputs = keras.applications.resnet50.preprocess_input(images_resized * 255)

# Make predictions
Y_proba = model.predict(inputs)

# Decode predictions
top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=3)
```

Available models: ResNet variants, GoogLeNet variants (Inception-v3, Xception), VGGNet, MobileNet, MobileNetV2.

## Transfer Learning

When building classifiers for classes not in ImageNet, reuse lower layers of pretrained models.

Example: Classifying flower images using pretrained Xception:

```python
# Load dataset
import tensorflow_datasets as tfds
dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples  # 3670
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes  # 5

# Split dataset
test_split, valid_split, train_split = tfds.Split.TRAIN.subsplit([10, 15, 75])
test_set = tfds.load("tf_flowers", split=test_split, as_supervised=True)
valid_set = tfds.load("tf_flowers", split=valid_split, as_supervised=True)
train_set = tfds.load("tf_flowers", split=train_split, as_supervised=True)

# Preprocessing function
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

# Apply preprocessing, shuffle, batch, prefetch
batch_size = 32
train_set = train_set.shuffle(1000).map(preprocess).batch(batch_size).prefetch(1)
valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)

# Build model
base_model = keras.applications.xception.Xception(weights="imagenet",
                                                  include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.Model(inputs=base_model.input, outputs=output)

# Freeze pretrained layers initially
for layer in base_model.layers:
    layer.trainable = False

# Compile and train
optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", 
             optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set, epochs=5, validation_data=valid_set)

# Unfreeze and fine-tune
for layer in base_model.layers:
    layer.trainable = True
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001)
model.compile(...)
history = model.fit(...)
```

Can achieve ~95% accuracy on test set.

## Classification and Localization

To locate objects, add regression task predicting bounding box coordinates:

```python
base_model = keras.applications.xception.Xception(weights="imagenet",
                                                  include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
class_output = keras.layers.Dense(n_classes, activation="softmax")(avg)
loc_output = keras.layers.Dense(4)(avg)  # center_x, center_y, height, width
model = keras.Model(inputs=base_model.input,
                   outputs=[class_output, loc_output])
model.compile(loss=["sparse_categorical_crossentropy", "mse"],
             loss_weights=[0.8, 0.2],
             optimizer=optimizer, metrics=["accuracy"])
```

**Requirements:**
- Bounding box labels for training data (use tools like VGG Image Annotator, LabelImg)
- Normalize coordinates to 0-1 range
- Often predict √(height) and √(width) instead of raw dimensions

**Evaluation metric: Intersection over Union (IoU)**
- Area of overlap / Area of union
- Implemented in `tf.keras.metrics.MeanIoU`

## Object Detection

Detecting and localizing multiple objects in an image.

### Traditional Approach: Sliding CNN

Slide a CNN across the image at different scales:
1. Divide image into grid
2. Slide CNN across all regions of various sizes
3. Apply **non-max suppression:**
   - Add objectness output (probability object is present)
   - Remove predictions with objectness < threshold
   - Keep bounding box with highest objectness score
   - Remove overlapping boxes (IoU > threshold, e.g., 0.6)
   - Repeat until done

**Problem:** Requires running CNN many times (slow).

### Fully Convolutional Networks (FCN)

Convert dense layers to convolutional layers:
- Dense layer with 200 neurons on top of 100×7×7 feature maps
- Replace with Conv2D with 200 filters, 7×7 kernel, "valid" padding
- Produces 200×1×1 output (same as dense layer but spatial)

**Key advantage:** Can process images of any size!

Example:
- CNN trained on 224×224 images with overall stride 32
- Output: 7×7 feature maps
- Convert to FCN and feed 448×448 image
- Output: 14×14 feature maps
- With 10 output filters (7×7, stride 1): 8×8 grid of predictions
- Like sliding CNN 8×8 times, but processes image only once!

### YOLO (You Only Look Once)

Extremely fast, real-time object detection.

Key features:
1. **Multiple bounding boxes per cell:** 5 boxes per grid cell (not just 1)
2. **Relative coordinates:** Predicts offset relative to grid cell (0,0 = top-left, 1,1 = bottom-right), using logistic activation
3. **Anchor boxes:** Uses K-Means to find 5 representative bounding box dimensions, predicts rescaling factors
4. **Multi-scale training:** Trains on images of different sizes (330×330 to 608×608)
5. **Skip connections:** Recovers spatial resolution lost in CNN

Each grid cell outputs:
- 5 bounding boxes (4 coordinates each) = 20 numbers
- 5 objectness scores = 5 numbers
- 20 class probabilities (PASCAL VOC dataset) = 20 numbers
- Total: 45 numbers per grid cell

**Extensions:**
- YOLOv2, YOLOv3 with incremental improvements
- YOLO9000 uses hierarchical classification (WordTree) for 9000+ classes

### Mean Average Precision (mAP)

Standard metric for object detection:

1. **Precision/Recall curve:** Plot precision vs. recall
2. **Average Precision (AP):** For each recall level (0%, 10%, 20%, ..., 100%), find maximum precision achievable at that recall. Average these maximum precisions.
3. **mAP:** Average AP across all classes

**Object detection specifics:**
- Prediction correct if IoU > threshold AND class correct
- **mAP@0.5:** Uses IoU threshold of 0.5
- **mAP@[.50:.95]:** Averages mAP over IoU thresholds from 0.50 to 0.95 (step 0.05)

**Other architectures:**
- **SSD (Single Shot Detector):** Similar to YOLO
- **Faster R-CNN:** Image → CNN → Region Proposal Network → Classifier for each proposed region

## Semantic Segmentation

Classify each pixel according to object class it belongs to.

**Challenge:** CNNs reduce spatial resolution (due to strides), losing precise location information.

### Basic Approach

1. Take pretrained CNN, convert to FCN
2. Add **transposed convolutional layer** (or deconvolution) for upsampling
3. Transposed convolution: Insert empty rows/columns, then convolve (equivalent to fractional stride)

```python
# Transposed convolution layer
upsampling = keras.layers.Conv2DTranspose(filters, kernel_size, 
                                          strides=2, padding="same")
```

**Note:** In transposed convolution, larger stride = larger output (opposite of regular convolution).

### Skip Connections for Better Resolution

To recover spatial resolution:
1. Upsample output by 2× (instead of 32×)
2. Add output from lower layer (double resolution)
3. Upsample result by 16×
4. Total upsampling: 32×

This recovers spatial information lost in pooling layers. Can add multiple skip connections from different depths.

**Super-resolution:** Can upsample beyond original image size to increase resolution.

### Instance Segmentation

Like semantic segmentation but distinguishes individual objects (not merging objects of same class).

**Mask R-CNN:** Extends Faster R-CNN by adding pixel mask prediction for each bounding box.

## Other TensorFlow Convolution Operations

```python
# 1D convolution (for time series, text)
keras.layers.Conv1D(...)

# 3D convolution (for 3D medical scans)
keras.layers.Conv3D(...)

# Dilated (à-trous) convolution
keras.layers.Conv2D(..., dilation_rate=2)  # Inserts holes in filter

# Depthwise convolution (manual)
tf.nn.depthwise_conv2d()  # Applies each filter to each channel independently
```

## Key Takeaways

1. **CNNs solve the parameter explosion** of fully connected networks for images
2. **Convolutional layers** detect features at any location via weight sharing
3. **Pooling layers** reduce dimensionality and provide some translation invariance
4. **Skip connections** (ResNet) enable training very deep networks
5. **Pretrained models** enable transfer learning for custom tasks
6. **Object detection** (YOLO, Faster R-CNN) locates multiple objects
7. **Semantic segmentation** classifies every pixel
8. **Data augmentation** is crucial for preventing overfitting

The field progresses rapidly with new architectures addressing adversarial learning, explainability, realistic image generation, one-shot learning, and novel approaches like capsule networks.

## Exercises

1. What advantages do CNNs have over fully connected DNNs for image classification?
2. For a CNN with three 3×3 convolutional layers (stride 2, "same" padding) outputting 100, 200, and 400 feature maps on 200×300 RGB images, calculate total parameters and RAM requirements.
3. What can you do if GPU runs out of memory training a CNN?
4. Why add max pooling instead of convolutional layer with same stride?
5. When to use local response normalization?
6. Name main innovations in AlexNet, GoogLeNet, ResNet, SENet, and Xception.
7. What is a fully convolutional network? How to convert dense to convolutional layer?
8. What is the main technical difficulty of semantic segmentation?
9. Build your own CNN for MNIST from scratch.
10. Use transfer learning for large image classification with your own dataset.
11. Try TensorFlow's Style Transfer tutorial.