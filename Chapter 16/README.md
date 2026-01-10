# Chapter 16: Natural Language Processing with RNNs and Attention

## Overview

This chapter explores Natural Language Processing (NLP) using Recurrent Neural Networks and attention mechanisms. We'll cover character-level and word-level models, sentiment analysis, neural machine translation, and the revolutionary Transformer architecture.

## 1. Generating Shakespearean Text Using Character RNN

### Creating the Training Dataset

Download and prepare Shakespeare's complete works:

```python
shakespeare_url = "https://homl.info/shakespeare"
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()
```

### Tokenization

Use Keras's `Tokenizer` for character-level encoding:

```python
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts([shakespeare_text])

# Encode text
[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
max_id = len(tokenizer.word_index)  # number of distinct characters
```

### Splitting Sequential Datasets

**Key Principles:**
- Avoid overlap between training, validation, and test sets
- Split across time (e.g., first 90% for training, next 5% for validation, 5% for test)
- Leave gaps between sets to avoid paragraph overlap
- For time series, ensure stationarity (patterns in training exist in future)

```python
train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
```

### Creating Windows

Convert long sequence into manageable windows using truncated backpropagation through time:

```python
n_steps = 100
window_length = n_steps + 1  # target = input shifted 1 character ahead
dataset = dataset.window(window_length, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))
```

**Figure 16-1 Summary:** Dataset preparation involves windowing, shuffling, batching, and separating inputs from targets.

### Complete Dataset Preparation

```python
batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
dataset = dataset.map(
    lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
dataset = dataset.prefetch(1)
```

### Building the Model

```python
model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id],
                     dropout=0.2, recurrent_dropout=0.2),
    keras.layers.GRU(128, return_sequences=True,
                     dropout=0.2, recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                     activation="softmax"))
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
history = model.fit(dataset, epochs=20)
```

### Generating Text

Use temperature to control diversity:

```python
def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model.predict(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]

def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text
```

**Temperature effects:**
- Close to 0: favors high-probability characters (conservative)
- High values: gives all characters equal probability (creative/random)
- Around 1: balanced diversity

## 2. Stateful RNN

### Key Concept

Preserve hidden state between training batches to learn longer patterns.

### Dataset Preparation

Use non-overlapping, sequential windows:

```python
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_length))
dataset = dataset.batch(1)  # Single window per batch
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
```

### Model Creation

```python
model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, stateful=True,
                     dropout=0.2, recurrent_dropout=0.2,
                     batch_input_shape=[batch_size, None, max_id]),
    keras.layers.GRU(128, return_sequences=True, stateful=True,
                     dropout=0.2, recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id,
                                                     activation="softmax"))
])
```

### Reset States Between Epochs

```python
class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
model.fit(dataset, epochs=50, callbacks=[ResetStatesCallback()])
```

**Note:** After training, copy weights to a stateless model for flexible inference.

## 3. Sentiment Analysis

### Loading IMDb Dataset

```python
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data()
```

Data is preprocessed: words → integers by frequency (0=pad, 1=start-of-sequence, 2=unknown).

### Modern Tokenization

Better alternatives to simple space-based splitting:
- **Subword tokenization** (SentencePiece): language-independent, handles unseen words
- **Byte Pair Encoding (BPE)**: creates subword encodings
- **WordPiece**: TensorFlow's variant in TF.Text library

### Preprocessing with TensorFlow Operations

```python
def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)  # Truncate to 300 chars
    X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch
```

### Building Vocabulary

```python
from collections import Counter

vocabulary = Counter()
for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
    for review in X_batch:
        vocabulary.update(list(review.numpy()))

vocab_size = 10000
truncated_vocabulary = [
    word for word, count in vocabulary.most_common()[:vocab_size]]
```

### Creating Lookup Table

```python
words = tf.constant(truncated_vocabulary)
word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
num_oov_buckets = 1000
table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)
```

### Building the Model

```python
embed_size = 128
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
                          input_shape=[None]),
    keras.layers.GRU(128, return_sequences=True),
    keras.layers.GRU(128),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="adam",
             metrics=["accuracy"])
```

### Masking

Enable masking to ignore padding tokens:

```python
keras.layers.Embedding(..., mask_zero=True)
```

**How it works:**
- Creates mask tensor: `K.not_equal(inputs, 0)`
- Automatically propagated through model
- Recurrent layers ignore masked time steps (copy previous output)
- Only works if time dimension is preserved

**Manual masking for complex models:**

```python
K = keras.backend
inputs = keras.layers.Input(shape=[None])
mask = keras.layers.Lambda(lambda inputs: K.not_equal(inputs, 0))(inputs)
z = keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size)(inputs)
z = keras.layers.GRU(128, return_sequences=True)(z, mask=mask)
z = keras.layers.GRU(128)(z, mask=mask)
outputs = keras.layers.Dense(1, activation="sigmoid")(z)
model = keras.Model(inputs=[inputs], outputs=[outputs])
```

### Reusing Pretrained Embeddings

Use TensorFlow Hub for pretrained embeddings:

```python
import tensorflow_hub as hub

model = keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1",
                   dtype=tf.string, input_shape=[], output_shape=[50]),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])
```

Benefits:
- Trained on massive corpora (e.g., Google News 7B)
- Better embeddings with fewer parameters
- Faster training

## 4. Neural Machine Translation (NMT)

### Encoder-Decoder Architecture

**Key components:**
1. **Encoder**: Processes input (English) sentence, outputs representations
2. **Decoder**: Generates output (French) translation, receives encoder outputs
3. Input sentences reversed for better gradient flow
4. Decoder receives previous target word as input (teacher forcing)

**Important considerations:**
- Variable-length sequences: use masking and bucketing
- Ignore outputs after EOS token
- Use sampled softmax for large vocabularies

### Implementation with TensorFlow Addons

```python
import tensorflow_addons as tfa

encoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
decoder_inputs = keras.layers.Input(shape=[None], dtype=np.int32)
sequence_lengths = keras.layers.Input(shape=[], dtype=np.int32)

embeddings = keras.layers.Embedding(vocab_size, embed_size)
encoder_embeddings = embeddings(encoder_inputs)
decoder_embeddings = embeddings(decoder_inputs)

encoder = keras.layers.LSTM(512, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embeddings)
encoder_state = [state_h, state_c]

sampler = tfa.seq2seq.sampler.TrainingSampler()
decoder_cell = keras.layers.LSTMCell(512)
output_layer = keras.layers.Dense(vocab_size)
decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler,
                                                  output_layer=output_layer)

final_outputs, final_state, final_sequence_lengths = decoder(
    decoder_embeddings, initial_state=encoder_state,
    sequence_length=sequence_lengths)

Y_proba = tf.nn.softmax(final_outputs.rnn_output)
model = keras.Model(inputs=[encoder_inputs, decoder_inputs, sequence_lengths],
                   outputs=[Y_proba])
```

### Bidirectional RNNs

Process sequences in both directions to capture context:

```python
keras.layers.Bidirectional(keras.layers.GRU(10, return_sequences=True))
```

Output size doubles (concatenates both directions).

### Beam Search

Keep track of k most promising sentences instead of greedy selection:

```python
beam_width = 10
decoder = tfa.seq2seq.beam_search_decoder.BeamSearchDecoder(
    cell=decoder_cell, beam_width=beam_width, output_layer=output_layer)

decoder_initial_state = tfa.seq2seq.beam_search_decoder.tile_batch(
    encoder_state, multiplier=beam_width)

outputs, _, _ = decoder(
    embedding_decoder, start_tokens=start_tokens, end_token=end_token,
    initial_state=decoder_initial_state)
```

**Benefits:** Allows model to "reconsider" early decisions, improves translation quality.

## 5. Attention Mechanisms

### Motivation

RNNs struggle with long sequences due to limited short-term memory. Attention creates shortcuts between encoder and decoder.

### Bahdanau Attention (Concatenative)

Decoder attends to relevant encoder outputs at each step:

**Weight calculation:**
```
α(t,i) = exp(e(t,i)) / Σ exp(e(t,i'))

where e(t,i) comes from alignment model
```

**Implementation concept:**
1. Concatenate encoder outputs with decoder's previous hidden state
2. Pass through Dense layer to get scores
3. Apply softmax to get weights α
4. Compute weighted sum of encoder outputs

### Luong Attention (Multiplicative)

Simpler, faster alternative using dot product:

**Three variants:**

```
Attention mechanisms:

c̃t = Σ α(t,i) * hi

with α(t,i) = exp(e(t,i)) / Σ exp(e(t,i'))

and e(t,i) = {
    h̃t^T * hi              (dot)
    h̃t^T * W * hi          (general)
    v^T * tanh(W[h̃t; hi])  (concat)
}
```

**TensorFlow Addons implementation:**

```python
attention_mechanism = tfa.seq2seq.attention_wrapper.LuongAttention(
    units, encoder_state, memory_sequence_length=encoder_sequence_length)

attention_decoder_cell = tfa.seq2seq.attention_wrapper.AttentionWrapper(
    decoder_cell, attention_mechanism, attention_layer_size=n_units)
```

### Visual Attention

Applied to image captioning: decoder attends to relevant image regions when generating each word.

### Explainability

Attention weights reveal what the model focuses on, enabling:
- Debugging model mistakes
- Understanding decision process
- Meeting regulatory requirements

## 6. The Transformer Architecture

### Overview

Revolutionary "Attention Is All You Need" (2017) architecture using only attention mechanisms—no recurrence or convolution.

**Key advantages:**
- Parallelizable (faster training)
- Better at capturing long-range dependencies
- State-of-the-art performance

### Architecture Components

**Encoder (left side):**
- Input: batch of word ID sequences
- Outputs: 512-dimensional representations
- Stacked N=6 times

**Decoder (right side):**
- Input: target sentence (shifted right) + encoder outputs
- Outputs: probability distribution over vocabulary
- Also stacked N=6 times

**Novel components:**
1. **Multi-Head Attention layers**
2. **Positional embeddings**
3. Skip connections + Layer Normalization
4. Feed Forward blocks (2 Dense layers)

### Positional Embeddings

Since attention doesn't consider word order, add position information:

```
Positional embedding formula:

P(p, 2i) = sin(p / 10000^(2i/d))
P(p, 2i+1) = cos(p / 10000^(2i/d))

where:
- p = position in sentence
- i = dimension index
- d = embedding dimension
```

**Properties:**
- Unique embedding for each position
- Model can learn relative positions
- Extends to arbitrary sentence lengths

**Implementation:**

```python
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, max_steps, max_dims, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        if max_dims % 2 == 1: max_dims += 1
        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / max_dims)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
    
    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]
```

### Scaled Dot-Product Attention

Core attention mechanism:

```
Attention(Q, K, V) = softmax(QK^T / √d_keys) * V

where:
- Q: queries [n_queries, d_keys]
- K: keys [n_keys, d_keys]
- V: values [n_keys, d_values]
- Scaling factor prevents gradient saturation
```

**Keras implementation:**

```python
Z = encoder_in
for N in range(6):
    Z = keras.layers.Attention(use_scale=True)([Z, Z])
encoder_outputs = Z

Z = decoder_in
for N in range(6):
    Z = keras.layers.Attention(use_scale=True, causal=True)([Z, Z])
    Z = keras.layers.Attention(use_scale=True)([Z, encoder_outputs])

outputs = keras.layers.TimeDistributed(
    keras.layers.Dense(vocab_size, activation="softmax"))(Z)
```

### Multi-Head Attention

Multiple attention layers in parallel, each focusing on different aspects:

**Intuition:**
- Single attention layer captures all features at once
- Multiple "heads" project into different subspaces
- Each head can focus on specific characteristics (verb, tense, etc.)
- Results concatenated and projected back

**Architecture:**
1. Apply h different linear transformations to Q, K, V
2. Compute h parallel Scaled Dot-Product Attention operations
3. Concatenate results
4. Apply final linear transformation

## 7. Recent Language Model Innovations (2018-2019)

### ELMo (Embeddings from Language Models)
- Contextualized word embeddings
- Same word has different embeddings in different contexts
- Uses internal states of deep bidirectional language model

### ULMFiT (Universal Language Model Fine-tuning)
- Demonstrated power of unsupervised pretraining
- LSTM trained on huge corpus via self-supervised learning
- Fine-tuned on specific tasks
- 100 labeled examples matched 10,000 from-scratch examples

### GPT (Generative Pre-trained Transformer)
- Transformer-based unsupervised pretraining
- 12 stacked Transformer modules
- Fine-tuned with minimal adaptations per task
- **GPT-2**: 1.5 billion parameters, zero-shot learning capability

### BERT (Bidirectional Encoder Representations from Transformers)

**Key innovation - Two pretraining tasks:**

1. **Masked Language Model (MLM)**
   - 15% of words randomly masked
   - Model predicts masked words
   - 80% actually masked, 10% random word, 10% unchanged

2. **Next Sentence Prediction (NSP)**
   - Predict if two sentences are consecutive
   - Significantly improves question answering and entailment tasks

**Architecture:**
- Non-masked Multi-Head Attention (bidirectional)
- Trained on huge corpus with self-supervision
- Fine-tuned with minimal changes

## Key Takeaways

1. **Character RNNs** can learn to generate text by predicting next character
2. **Stateful RNNs** preserve hidden state to learn longer patterns
3. **Masking** efficiently handles variable-length sequences
4. **Pretrained embeddings** boost performance with less data
5. **Encoder-Decoder** architecture enables sequence-to-sequence tasks
6. **Attention mechanisms** solve RNN memory limitations
7. **Transformers** achieve state-of-the-art with parallelizable attention-only architecture
8. **Pretraining + fine-tuning** is the dominant paradigm for NLP

## Practical Tips

- Use temperature ~1 for text generation diversity
- Implement beam search for better translations
- Leverage pretrained models from TF Hub
- Add positional encodings when using attention without recurrence
- Use masking to ignore padding tokens
- Consider subword tokenization for better generalization
- Pretrain on large corpora, fine-tune on specific tasks

## Future Directions

The field evolves rapidly. While Transformers dominate currently, alternatives emerge:
- 2D CNNs for sequence-to-sequence (masked convolutions)
- Independent RNNs (IndRNN) for deeper, longer-sequence models
- Hybrid architectures combining strengths of different approaches
