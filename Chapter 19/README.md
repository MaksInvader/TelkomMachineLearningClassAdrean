# Chapter 19: Training and Deploying TensorFlow Models at Scale

## Overview

This chapter covers deploying TensorFlow models to production and scaling training across multiple devices. Key topics include:
- Serving models with TF Serving
- Deploying to Google Cloud AI Platform
- Mobile and embedded deployment
- GPU acceleration
- Distributed training strategies

## Serving a TensorFlow Model

### Using TensorFlow Serving

TF Serving is an efficient, battle-tested C++ model server that can:
- Sustain high load
- Serve multiple model versions
- Automatically deploy latest versions
- Handle graceful model transitions

### Exporting SavedModels

Export models to TensorFlow's SavedModel format:

```python
model = keras.models.Sequential([...])
model.compile([...])
history = model.fit([...])

model_version = "0001"
model_name = "my_mnist_model"
model_path = os.path.join(model_name, model_version)
tf.saved_model.save(model, model_path)
```

**SavedModel Directory Structure:**
```
my_mnist_model
└── 0001
    ├── assets
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```

**Best Practice:** Include preprocessing layers in the final model to simplify deployment.

**Limitation:** SavedModels only work with TensorFlow operations (excludes `tf.py_function()` and dynamic Keras models).

### Inspecting SavedModels

Use the command-line tool:

```bash
$ saved_model_cli show --dir my_mnist_model/0001 --all
```

Load a SavedModel:

```python
# As SavedModel object
saved_model = tf.saved_model.load(model_path)
y_pred = saved_model(tf.constant(X_new, dtype=tf.float32))

# As Keras model
model = keras.models.load_model(model_path)
y_pred = model.predict(tf.constant(X_new, dtype=tf.float32))
```

### Installing TensorFlow Serving

Using Docker (recommended):

```bash
# Pull the image
$ docker pull tensorflow/serving

# Run the container
$ docker run -it --rm -p 8500:8500 -p 8501:8501 \
  -v "$ML_PATH/my_mnist_model:/models/my_mnist_model" \
  -e MODEL_NAME=my_mnist_model \
  tensorflow/serving
```

**Key Docker Options:**
- `-it`: Interactive mode
- `--rm`: Delete container when stopped
- `-p 8500:8500`: Forward gRPC port
- `-p 8501:8501`: Forward REST API port
- `-v`: Mount model directory
- `-e MODEL_NAME`: Set model name

### Querying via REST API

```python
import json
import requests

# Prepare request
input_data_json = json.dumps({
    "signature_name": "serving_default",
    "instances": X_new.tolist(),
})

# Send request
SERVER_URL = 'http://localhost:8501/v1/models/my_mnist_model:predict'
response = requests.post(SERVER_URL, data=input_data_json)
response.raise_for_status()
response = response.json()

# Extract predictions
y_proba = np.array(response["predictions"])
```

**Limitation:** REST uses JSON (text-based), which is verbose and inefficient for large data transfers.

### Querying via gRPC API

gRPC is more efficient (compact binary format, HTTP/2):

```python
from tensorflow_serving.apis.predict_pb2 import PredictRequest
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

# Create request
request = PredictRequest()
request.model_spec.name = model_name
request.model_spec.signature_name = "serving_default"
input_name = model.input_names[0]
request.inputs[input_name].CopyFrom(tf.make_tensor_proto(X_new))

# Send request
channel = grpc.insecure_channel('localhost:8500')
predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
response = predict_service.Predict(request, timeout=10.0)

# Extract predictions
output_name = model.output_names[0]
outputs_proto = response.outputs[output_name]
y_proba = tf.make_ndarray(outputs_proto)
```

**Recommendation:** Use gRPC for large data transfers due to better performance.

### Deploying New Model Versions

Simply export to a new version directory (e.g., `0002`). TF Serving automatically:
1. Detects the new version
2. Handles pending requests with old version
3. Switches to new version for new requests
4. Unloads old version

**Automatic Batching:** Enable with `--enable_batching` to batch multiple requests together for better GPU utilization (trades latency for throughput).

## Deploying to Google Cloud AI Platform

### Setup Steps

1. **Create GCP account** and enable billing
2. **Create/select project**
3. **Create GCS bucket** for storing models
4. **Upload SavedModel** to GCS bucket
5. **Create model** on AI Platform
6. **Create model version** specifying:
   - Python version (3.5+)
   - TensorFlow version
   - Machine type
   - Model path on GCS
   - Scaling options (automatic/manual)

### Using the Prediction Service

**Create service account** for authentication:
1. Go to IAM & admin → Service accounts
2. Create account with ML Engine Developer role
3. Export private key as JSON

**Set credentials:**

```python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "my_service_account_key.json"
```

**Query the service:**

```python
import googleapiclient.discovery

project_id = "your-project-id"
model_id = "my_mnist_model"
model_path = f"projects/{project_id}/models/{model_id}"
ml_resource = googleapiclient.discovery.build("ml", "v1").projects()

def predict(X):
    input_data_json = {
        "signature_name": "serving_default",
        "instances": X.tolist()
    }
    request = ml_resource.predict(name=model_path, body=input_data_json)
    response = request.execute()
    if "error" in response:
        raise RuntimeError(response["error"])
    return np.array([pred[output_name] for pred in response["predictions"]])

Y_probas = predict(X_new)
```

**Benefits:**
- Automatic scaling
- No infrastructure management
- Pay only for usage
- Monitoring via Stackdriver

## Mobile and Embedded Deployment

### TensorFlow Lite (TFLite)

**Objectives:**
1. Reduce model size (faster download, less RAM)
2. Reduce computations (lower latency, less battery drain)
3. Adapt to device constraints

### Model Conversion

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model = converter.convert()
with open("converted_model.tflite", "wb") as f:
    f.write(tflite_model)
```

**Optimizations:**
- Uses FlatBuffers format (efficient serialization)
- Prunes unused operations
- Optimizes computations (fuses operations)
- Can convert from SavedModel or Keras model directly

### Quantization

**Post-training quantization (weights only):**

```python
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
```

**Benefits:**
- 4× size reduction (32-bit floats → 8-bit integers)
- Symmetrical quantization: maps float range [−m, +m] to integer range [−127, +127]

**Example:** If weights range from −1.5 to +0.8:
- −127 → −1.5
- 0 → 0.0
- +127 → +1.5

**Limitations:** Weights are converted back to floats at runtime (no speed improvement, no RAM reduction).

**Full quantization (weights + activations):**
- Requires calibration with representative data
- Enables integer-only inference
- Necessary for some accelerators (e.g., Edge TPU)

**Quantization-aware training:** Add fake quantization operations during training to make weights more robust to quantization noise.

## TensorFlow in the Browser

### TensorFlow.js

Use cases:
- Offline functionality
- Low latency (no server round-trip)
- Privacy (data stays on client)

**Export model:**

```bash
tensorflowjs_converter --input_format=tf_saved_model \
  saved_model_dir output_dir
```

**Load and use in browser:**

```javascript
import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('https://example.com/tfjs/model.json');
const image = tf.fromPixels(webcamElement);
const prediction = model.predict(image);
```

## Using GPUs

### Getting a GPU

**Options:**
1. Purchase Nvidia GPU card (requires CUDA Compute Capability 3.5+)
2. Use cloud GPU VMs (GCP, AWS, Azure)
3. Use Colaboratory (free!)

**Required installations:**
- Nvidia drivers
- CUDA library
- cuDNN library

### Verification

```bash
# Check GPU availability
$ nvidia-smi
```

```python
# In Python
import tensorflow as tf
tf.test.is_gpu_available()  # True
tf.test.gpu_device_name()  # '/device:GPU:0'
tf.config.experimental.list_physical_devices(device_type='GPU')
```

### Google Colaboratory

**Setup:**
1. Go to https://colab.research.google.com/
2. Create new notebook
3. Runtime → Change runtime type → GPU

**Limitations:**
- Max 5 notebooks per runtime type
- ~30 min idle timeout
- 12-hour maximum runtime
- No cryptocurrency mining

### Managing GPU RAM

**Option 1: Assign GPUs to processes**

```bash
$ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 python3 program_1.py
$ CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,2 python3 program_2.py
```

**Option 2: Set memory limit**

```python
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
    )
```

**Option 3: Dynamic memory growth**

```python
for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)
```

Or set environment variable: `TF_FORCE_GPU_ALLOW_GROWTH=true`

**Option 4: Split GPU into virtual devices**

```python
physical_gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_virtual_device_configuration(
    physical_gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
)
```

### Placing Operations on Devices

**Default behavior:**
- Variables and operations → GPU 0
- Operations without GPU kernels → CPU 0

**Manual placement:**

```python
with tf.device("/cpu:0"):
    c = tf.Variable(42.0)
```

**Soft placement:** Fall back to CPU if GPU unavailable:

```python
tf.config.set_soft_device_placement(True)
```

### Parallel Execution

TensorFlow automatically parallelizes operations:

**CPU:**
- **Inter-op thread pool:** Executes independent operations in parallel
- **Intra-op thread pool:** Executes multi-threaded kernels

**GPU:**
- Operations execute sequentially
- Each operation uses multi-threaded kernel

**Control thread counts:**

```python
tf.config.threading.set_inter_op_parallelism_threads(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
```

## Distributed Training

### Model Parallelism

Split model across devices:
- **Fully connected networks:** Hard to parallelize efficiently
- **CNNs:** Easier with partially connected layers
- **RNNs:** Horizontal splitting possible but communication overhead high

**Conclusion:** Model parallelism requires careful design and is architecture-specific.

### Data Parallelism

Replicate model across devices, train on different data subsets:

**Mirrored Strategy:**
- Mirror all parameters across GPUs
- Synchronous updates
- Use AllReduce algorithm to compute mean gradients

**Centralized Parameters:**
- Store parameters on CPU or parameter servers
- Can be synchronous or asynchronous

**Synchronous updates:**
- Wait for all replicas
- Can use spare replicas (~10%) to avoid slowest workers
- More stable convergence

**Asynchronous updates:**
- No waiting between replicas
- Faster but may suffer from stale gradients
- Mitigation: reduce learning rate, drop/scale stale gradients, adjust batch size

**Bandwidth Saturation:**
- Adding more GPUs eventually hurts performance
- Typical speedups: 25-40× on 50 GPUs (dense), 300× on 500 GPUs (sparse)
- Solutions: fewer powerful GPUs, better interconnects, reduce precision (float32 → bfloat16), shard parameters

## Distribution Strategies API

### Mirrored Strategy (Single Machine)

```python
distribution = tf.distribute.MirroredStrategy()
with distribution.scope():
    mirrored_model = keras.models.Sequential([...])
    mirrored_model.compile([...])

batch_size = 100  # Must be divisible by number of replicas
history = mirrored_model.fit(X_train, y_train, epochs=10)
```

**Specify devices:**

```python
distribution = tf.distribute.MirroredStrategy(["/gpu:0", "/gpu:1"])
```

**AllReduce implementations:**
- NCCL (default, usually fastest)
- HierarchicalCopyAllReduce
- ReductionToOneDevice

### Central Storage Strategy

```python
distribution = tf.distribute.experimental.CentralStorageStrategy()
```

Parameters stored on CPU (or GPU if only one), supports synchronous/asynchronous updates.

### Multi-Worker Mirrored Strategy

**Setup TF_CONFIG:**

```python
import os
import json

cluster_spec = {
    "worker": [
        "machine-a.example.com:2222",  # /job:worker/task:0
        "machine-b.example.com:2222"   # /job:worker/task:1
    ]
}

os.environ["TF_CONFIG"] = json.dumps({
    "cluster": cluster_spec,
    "task": {"type": "worker", "index": 0}
})
```

**Training code:**

```python
distribution = tf.distribute.experimental.MultiWorkerMirroredStrategy()
with distribution.scope():
    mirrored_model = keras.models.Sequential([...])
    mirrored_model.compile([...])

batch_size = 100
history = mirrored_model.fit(X_train, y_train, epochs=10)
```

**Cluster roles:**
- **worker:** Performs computations
- **chief:** Worker + extra duties (logging, checkpoints)
- **ps (parameter server):** Manages parameters (ParameterServerStrategy only)
- **evaluator:** Handles evaluation

### Parameter Server Strategy

```python
distribution = tf.distribute.experimental.ParameterServerStrategy()
```

Add parameter servers to cluster, configure TF_CONFIG appropriately.

### TPU Strategy

```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.tpu.experimental.initialize_tpu_system(resolver)
tpu_strategy = tf.distribute.experimental.TPUStrategy(resolver)
```

## Training on Google Cloud AI Platform

### Submit Training Job

```bash
$ gcloud ai-platform jobs submit training my_job_20190531_164700 \
  --region asia-southeast1 \
  --scale-tier PREMIUM_1 \
  --runtime-version 2.0 \
  --python-version 3.5 \
  --package-path /my_project/src/trainer \
  --module-name trainer.task \
  --staging-bucket gs://my-staging-bucket \
  --job-dir gs://my-mnist-model-bucket/trained_model \
  -- \
  --my-extra-argument1 foo --my-extra-argument2 bar
```

**Scale tiers:** PREMIUM_1 = 20 workers + 11 parameter servers

**Access GCS data:**

```python
dataset = tf.data.TFRecordDataset("gs://my-data-bucket/my_data_001.tfrecord")
```

### Hyperparameter Tuning

**Create config file (tuning.yaml):**

```yaml
trainingInput:
  hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: accuracy
    maxTrials: 10
    maxParallelTrials: 2
    params:
      - parameterName: n_layers
        type: INTEGER
        minValue: 10
        maxValue: 100
        scaleType: UNIT_LINEAR_SCALE
      - parameterName: momentum
        type: DOUBLE
        minValue: 0.1
        maxValue: 1.0
        scaleType: UNIT_LOG_SCALE
```

**Submit job:**

```bash
$ gcloud ai-platform jobs submit training my_tuning_job \
  --config tuning.yaml \
  ...
```

**Scale types:**
- `UNIT_LINEAR_SCALE`: Flat prior
- `UNIT_LOG_SCALE`: Optimal value closer to max
- `UNIT_REVERSE_LOG_SCALE`: Optimal value closer to min

AI Platform uses **Google Vizier** (Bayesian optimization) and reads metrics from TensorBoard event files.

## Key Takeaways

1. **TF Serving** provides production-ready model serving with versioning and auto-deployment
2. **REST API** is simple but inefficient; **gRPC** is better for large data
3. **TFLite** optimizes models for mobile/embedded via compression and quantization
4. **GPUs** dramatically speed up training; manage RAM carefully
5. **Data parallelism** generally preferred over model parallelism
6. **Distribution Strategies API** simplifies distributed training
7. **Mirrored Strategy** best for single-machine multi-GPU
8. **Cloud platforms** (GCP AI Platform) handle infrastructure at scale
9. **Hyperparameter tuning** can be automated with Bayesian optimization

## Exercises

1. What does a SavedModel contain? How do you inspect its content?
2. When should you use TF Serving? What are its main features? What are some tools you can use to deploy it?
3. How do you deploy a model across multiple TF Serving instances?
4. When should you use the gRPC API rather than the REST API to query a model served by TF Serving?
5. What are the different ways TFLite reduces a model's size to make it run on a mobile or embedded device?
6. What is quantization-aware training, and why would you need it?
7. What are model parallelism and data parallelism? Why is the latter generally recommended?
8. When training a model across multiple servers, what distribution strategies can you use? How do you choose which one to use?
9. Train a model and deploy it to TF Serving or Google Cloud AI Platform. Write client code to query it using REST or gRPC API. Update the model and deploy the new version. Roll back to the first version.
10. Train any model across multiple GPUs using MirroredStrategy. Compare with CentralStorageStrategy.
11. Train a small model on Google Cloud AI Platform using black box hyperparameter tuning.

---

*"My greatest hope is that this book will inspire you to build a wonderful ML application that will benefit all of us!"* — Aurélien Géron