# Chapter 13: Loading and Preprocessing Data with TensorFlow

## Overview

Deep Learning systems often train on datasets too large for RAM. TensorFlow's Data API simplifies ingesting and preprocessing large datasets through dataset objects that handle multithreading, queuing, batching, and prefetching seamlessly with tf.keras.

The Data API reads from:
- Text files (CSV)
- Binary files with fixed-size records
- TFRecord format (flexible binary format with protocol buffers)
- SQL databases
- Various sources via open source extensions

## The Data API

### Creating Datasets

The Data API centers on the dataset concept - a sequence of data items.

```python
>>> X = tf.range(10)  # any data tensor
>>> dataset = tf.data.Dataset.from_tensor_slices(X)
>>> dataset
<TensorSliceDataset shapes: (), types: tf.int32>
```

Iterate over items:

```python
>>> for item in dataset:
...     print(item)
...
tf.Tensor(0, shape=(), dtype=int32)
tf.Tensor(1, shape=(), dtype=int32)
[...]
tf.Tensor(9, shape=(), dtype=int32)
```

### Chaining Transformations

Methods return new datasets for chaining:

```python
>>> dataset = dataset.repeat(3).batch(7)
>>> for item in dataset:
...     print(item)
...
tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int32)
tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int32)
tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int32)
tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int32)
tf.Tensor([8 9], shape=(2,), dtype=int32)
```

**Important:** Dataset methods create new datasets - keep references with `dataset = ...`

Transform items with `map()`:

```python
>>> dataset = dataset.map(lambda x: x * 2)  # Items: [0,2,4,6,8,10,12]
```

Set `num_parallel_calls` for multithreading on intensive operations.

The `apply()` method transforms the entire dataset:

```python
>>> dataset = dataset.apply(tf.data.experimental.unbatch())
```

Filter and limit datasets:

```python
>>> dataset = dataset.filter(lambda x: x < 10)  # Items: 0 2 4 6 8 0 2 4 6...
>>> for item in dataset.take(3):
...     print(item)
```

### Shuffling the Data

Gradient Descent requires independent, identically distributed instances. Use `shuffle()`:

```python
>>> dataset = tf.data.Dataset.range(10).repeat(3)
>>> dataset = dataset.shuffle(buffer_size=5, seed=42).batch(7)
```

The buffer size must be large enough for effective shuffling but not exceed RAM. For large datasets:
1. Shuffle source data itself
2. Split into multiple files, read randomly
3. Read multiple files simultaneously with interleaving
4. Add shuffling buffer with `shuffle()`

### Interleaving Files

Split data into CSV files:

```
MedInc,HouseAge,AveRooms,AveBedrms,Popul,AveOccup,Lat,Long,MedianHouseValue
3.5214,15.0,3.0499,1.1065,1447.0,1.6059,37.63,-122.43,1.442
5.3275,5.0,6.4900,0.9910,3464.0,3.4433,33.69,-117.39,1.687
[...]
```

Create filepath dataset:

```python
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)
```

Interleave files:

```python
n_readers = 5
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
    cycle_length=n_readers,
    num_parallel_calls=n_readers)
```

Set `num_parallel_calls=tf.data.experimental.AUTOTUNE` for dynamic threading.

### Preprocessing the Data

Implement preprocessing function:

```python
X_mean, X_std = [...]  # precomputed statistics
n_inputs = 8

def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / (X_std), y
```

**Key points:**
- `tf.io.decode_csv()` parses lines with default values defining column types
- `tf.stack()` converts scalar tensors to 1D arrays
- Apply standardization: `(x - mean) / std`

### Putting Everything Together

Helper function for efficient data loading:

```python
def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.shuffle(shuffle_buffer_size).repeat(repeat)
    return dataset.batch(batch_size).prefetch(1)
```

### Prefetching

`prefetch(1)` keeps the dataset one batch ahead - while training runs on one batch, the dataset prepares the next. This maximizes GPU utilization.

With multithreaded loading/preprocessing (via `num_parallel_calls`), CPU and GPU work in parallel for near 100% GPU utilization.

**Performance tip:** For small datasets fitting in memory, use `cache()` after loading/preprocessing but before shuffling to read each instance only once.

### Using with tf.keras

Create and use datasets:

```python
train_set = csv_reader_dataset(train_filepaths)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)

model = keras.models.Sequential([...])
model.compile([...])
model.fit(train_set, epochs=10, validation_data=valid_set)
model.evaluate(test_set)

new_set = test_set.take(3).map(lambda X, y: X)
model.predict(new_set)
```

Custom training loop:

```python
for X_batch, y_batch in train_set:
    [...]  # perform Gradient Descent step
```

Or create a TF Function for the entire loop:

```python
@tf.function
def train(model, optimizer, loss_fn, n_epochs, [...]):
    train_set = csv_reader_dataset(train_filepaths, repeat=n_epochs, [...])
    for X_batch, y_batch in train_set:
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

## The TFRecord Format

TFRecord is TensorFlow's preferred format for large datasets - a simple binary format containing variable-size records (length + CRC checksum + data + CRC checksum).

### Writing TFRecords

```python
with tf.io.TFRecordWriter("my_data.tfrecord") as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")
```

### Reading TFRecords

```python
filepaths = ["my_data.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item)
```

Output:
```
tf.Tensor(b'This is the first record', shape=(), dtype=string)
tf.Tensor(b'And this is the second record', shape=(), dtype=string)
```

Set `num_parallel_reads` for parallel file reading.

### Compressed TFRecords

```python
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
    [...]

dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"],
                                 compression_type="GZIP")
```

### Protocol Buffers

TFRecords usually contain serialized protocol buffers (protobufs) - portable, extensible binary format developed by Google.

Definition example:

```proto
syntax = "proto3";
message Person {
    string name = 1;
    int32 id = 2;
    repeated string email = 3;
}
```

Usage:

```python
>>> from person_pb2 import Person
>>> person = Person(name="Al", id=123, email=["a@b.com"])
>>> person.name = "Alice"
>>> person.email.append("c@d.com")
>>> s = person.SerializeToString()
>>> person2 = Person()
>>> person2.ParseFromString(s)
>>> person == person2
True
```

### TensorFlow Protobufs

The `Example` protobuf represents one dataset instance:

```proto
syntax = "proto3";
message BytesList { repeated bytes value = 1; }
message FloatList { repeated float value = 1 [packed = true]; }
message Int64List { repeated int64 value = 1 [packed = true]; }
message Feature {
    oneof kind {
        BytesList bytes_list = 1;
        FloatList float_list = 2;
        Int64List int64_list = 3;
    }
};
message Features { map<string, Feature> feature = 1; };
message Example { Features features = 1; };
```

Creating and writing an Example:

```python
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example

person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com", b"c@d.com"]))
        }))

with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
    f.write(person_example.SerializeToString())
```

### Loading and Parsing Examples

```python
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string),
}

for serialized_example in tf.data.TFRecordDataset(["my_contacts.tfrecord"]):
    parsed_example = tf.io.parse_single_example(serialized_example,
                                                feature_description)
```

Variable-length features return sparse tensors:

```python
>>> tf.sparse.to_dense(parsed_example["emails"], default_value=b"")
<tf.Tensor: [...] numpy=array([b'a@b.com', b'c@d.com'], [...])>
>>> parsed_example["emails"].values
<tf.Tensor: [...] numpy=array([b'a@b.com', b'c@d.com'], [...])>
```

**Special encodings:**
- Images: `tf.io.encode_jpeg()` → `tf.io.decode_jpeg()`
- Tensors: `tf.io.serialize_tensor()` → `tf.io.parse_tensor()`

Batch parsing:

```python
dataset = tf.data.TFRecordDataset(["my_contacts.tfrecord"]).batch(10)
for serialized_examples in dataset:
    parsed_examples = tf.io.parse_example(serialized_examples,
                                         feature_description)
```

### SequenceExample Protobuf

For lists of lists (e.g., documents with sentences):

```proto
message FeatureList { repeated Feature feature = 1; };
message FeatureLists { map<string, FeatureList> feature_list = 1; };
message SequenceExample {
    Features context = 1;
    FeatureLists feature_lists = 2;
};
```

Parse with `tf.io.parse_single_sequence_example()` or `tf.io.parse_sequence_example()`. Convert to ragged tensors:

```python
parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
    serialized_sequence_example, context_feature_descriptions,
    sequence_feature_descriptions)
parsed_content = tf.RaggedTensor.from_sparse(parsed_feature_lists["content"])
```

## Preprocessing Input Features

### Standardization Layer

Simple Lambda layer:

```python
means = np.mean(X_train, axis=0, keepdims=True)
stds = np.std(X_train, axis=0, keepdims=True)
eps = keras.backend.epsilon()

model = keras.models.Sequential([
    keras.layers.Lambda(lambda inputs: (inputs - means) / (stds + eps)),
    [...]  # other layers
])
```

Custom reusable layer:

```python
class Standardization(keras.layers.Layer):
    def adapt(self, data_sample):
        self.means_ = np.mean(data_sample, axis=0, keepdims=True)
        self.stds_ = np.std(data_sample, axis=0, keepdims=True)
    
    def call(self, inputs):
        return (inputs - self.means_) / (self.stds_ + keras.backend.epsilon())

std_layer = Standardization()
std_layer.adapt(data_sample)
model = keras.Sequential([std_layer, ...])
```

**Keras provides:** `keras.layers.Normalization` layer with similar functionality.

### Encoding Categorical Features

#### One-Hot Encoding

Create lookup table:

```python
vocab = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
indices = tf.range(len(vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
num_oov_buckets = 2
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
```

**Out-of-vocabulary (OOV) buckets** handle unknown categories via hashing. Use more buckets for more unknowns to avoid collisions.

Encode to one-hot:

```python
>>> categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
>>> cat_indices = table.lookup(categories)
>>> cat_indices
<tf.Tensor: [...] numpy=array([3, 5, 1, 1])>
>>> cat_one_hot = tf.one_hot(cat_indices, depth=len(vocab) + num_oov_buckets)
>>> cat_one_hot
<tf.Tensor: [...] numpy=
array([[0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0.],
       [0., 1., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0.]], dtype=float32)>
```

**Keras provides:** `keras.layers.TextVectorization` layer extracts vocabulary via `adapt()` and outputs word indices.

**Rule of thumb:**
- <10 categories → one-hot encoding
- >50 categories → embeddings
- 10-50 categories → experiment with both

#### Embeddings

An embedding is a trainable dense vector representing a category. Initially random, embeddings improve during training through representation learning.

Manual implementation:

```python
embedding_dim = 2
embed_init = tf.random.uniform([len(vocab) + num_oov_buckets, embedding_dim])
embedding_matrix = tf.Variable(embed_init)

>>> categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
>>> cat_indices = table.lookup(categories)
>>> tf.nn.embedding_lookup(embedding_matrix, cat_indices)
<tf.Tensor: [...] numpy=
array([[0.74011743, 0.8724445 ],
       [0.3103881 , 0.7223358 ],
       [0.3528825 , 0.46448255],
       [0.3528825 , 0.46448255]], dtype=float32)>
```

Using Keras:

```python
>>> embedding = keras.layers.Embedding(input_dim=len(vocab) + num_oov_buckets,
...                                   output_dim=embedding_dim)
>>> embedding(cat_indices)
```

Complete model with embeddings:

```python
regular_inputs = keras.layers.Input(shape=[8])
categories = keras.layers.Input(shape=[], dtype=tf.string)
cat_indices = keras.layers.Lambda(lambda cats: table.lookup(cats))(categories)
cat_embed = keras.layers.Embedding(input_dim=6, output_dim=2)(cat_indices)
encoded_inputs = keras.layers.concatenate([regular_inputs, cat_embed])
outputs = keras.layers.Dense(1)(encoded_inputs)
model = keras.models.Model(inputs=[regular_inputs, categories],
                          outputs=[outputs])
```

**Note:** One-hot encoding + Dense layer (no activation/bias) ≡ Embedding layer, but Embedding is more efficient.

#### Word Embeddings

Word embeddings (vectors representing words) have been used since the 1960s. In 2013, Tomáš Mikolov et al. trained neural networks to predict nearby words, obtaining remarkable embeddings where:
- Synonyms have close embeddings
- Semantically related words cluster together
- Meaningful axes emerge: King - Man + Woman ≈ Queen

**Warning:** Embeddings can capture biases (e.g., gender stereotypes). Ensuring fairness in Deep Learning is an active research area.

**Embedding dimensions:** Typically 10-300, depending on task and vocabulary size.

## Keras Preprocessing Layers

Standard preprocessing layers (may require checking current API):

### Normalization Layer

```python
normalization = keras.layers.Normalization()
normalization.adapt(data_sample)
# Use in model
```

### TextVectorization Layer

Encodes words to vocabulary indices:

```python
text_vectorization = keras.layers.TextVectorization()
text_vectorization.adapt(text_sample)
# Use in model
```

Options include:
- Word indices
- Word-count vectors (bag-of-words)
- TF-IDF (Term-Frequency × Inverse-Document-Frequency)

TF-IDF formula: word_count / log(total_documents_containing_word)

### Discretization Layer

Bins continuous data into categories with one-hot encoding:

```python
discretization = keras.layers.Discretization([...])
discretization.adapt(data_sample)
```

**Note:** Discretization is non-differentiable, use only at model start. Preprocessing layers are frozen during training.

### PreprocessingStage

Chain multiple preprocessing layers:

```python
normalization = keras.layers.Normalization()
discretization = keras.layers.Discretization([...])
pipeline = keras.layers.PreprocessingStage([normalization, discretization])
pipeline.adapt(data_sample)
```

### Custom Preprocessing Layers

Subclass `keras.layers.PreprocessingLayer`:

```python
class CustomPreprocessing(keras.layers.PreprocessingLayer):
    def adapt(self, data_sample, reset_state=True):
        if reset_state:
            # Reset state
        # Compute state from data_sample
    
    def call(self, inputs):
        # Apply preprocessing
```

## TF Transform

For computationally expensive preprocessing, process data once before training rather than per epoch. However, this creates training/serving skew - preprocessing differs in training vs production.

**TF Transform solution:** Define preprocessing once, use everywhere.

### Usage

Define preprocessing:

```python
import tensorflow_transform as tft

def preprocess(inputs):
    median_age = inputs["housing_median_age"]
    ocean_proximity = inputs["ocean_proximity"]
    standardized_age = tft.scale_to_z_score(median_age)
    ocean_proximity_id = tft.compute_and_apply_vocabulary(ocean_proximity)
    return {
        "standardized_median_age": standardized_age,
        "ocean_proximity_id": ocean_proximity_id
    }
```

TF Transform:
1. Applies `preprocess()` to training set via Apache Beam
2. Computes statistics (analyzers) over full dataset
3. Generates equivalent TensorFlow Function with computed constants
4. TF Function plugs into deployed model

Benefits:
- Preprocess once using Apache Beam/Spark
- Single preprocessing definition
- No training/serving skew
- Portable to mobile, web, etc.

## TensorFlow Datasets (TFDS)

Easy access to common datasets (MNIST, Fashion MNIST, ImageNet, etc.). Visit https://homl.info/tfds for full list.

### Usage

```python
import tensorflow_datasets as tfds

dataset = tfds.load(name="mnist")
mnist_train, mnist_test = dataset["train"], dataset["test"]

mnist_train = mnist_train.shuffle(10000).batch(32).prefetch(1)
for item in mnist_train:
    images = item["image"]
    labels = item["label"]
    [...]
```

Simplified for Keras:

```python
dataset = tfds.load(name="mnist", batch_size=32, as_supervised=True)
mnist_train = dataset["train"].prefetch(1)

model = keras.models.Sequential([...])
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")
model.fit(mnist_train, epochs=5)
```

Setting `as_supervised=True` returns tuples (features, labels) instead of dictionaries.

## Key Takeaways

1. **Data API**: Efficiently load and preprocess large datasets with multithreading, batching, and prefetching
2. **TFRecords**: Preferred format for large datasets, uses protocol buffers
3. **Preprocessing options**:
   - On-the-fly with Data API
   - In model with Keras layers
   - Ahead of time with TF Transform
4. **Categorical encoding**: One-hot for <10 categories, embeddings for >50
5. **Performance**: Use `prefetch()`, parallel processing, and `cache()` for small datasets
6. **TFDS**: Quick access to standard datasets

## Exercises

1. Why use the Data API?
2. Benefits of splitting datasets into multiple files?
3. How to identify and fix input pipeline bottlenecks?
4. Can TFRecords store any binary data?
5. Why convert to Example protobuf format?
6. When to compress TFRecords?
7. Compare preprocessing options (file writing, tf.data pipeline, model layers, TF Transform)
8. Common techniques for encoding categorical features and text?
9. Load Fashion MNIST, split into sets, save as TFRecords with serialized Examples, create efficient datasets, train with preprocessing layer
10. Download Large Movie Review Dataset, split, create efficient dataset, build binary classifier with TextVectorization and Embedding layers