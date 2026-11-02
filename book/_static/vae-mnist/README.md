# MNIST VAE: frozen training replay

This asset is a real VAE trained for this teaching book, rather than an image
atlas or an interpolation of cached output images. Only the decoder is evaluated
in the browser. The displayed originals are held-out MNIST images; their encoder
means and standard deviations were evaluated offline at each checkpoint.

## Model and run

- Training set: all **18,516** official MNIST training examples labelled 0, 1 or 8.
- Display examples: 30 images of each digit from the official **test** split.
  No digit labels are used in training; colors only identify examples afterwards.
- Encoder: 784 → 64 ReLU → separate two-dimensional mean and log-variance heads.
- Reparameterization: z = μ(x) + exp(log σ²(x)/2) ε, ε ∼ N(0,I).
- Decoder: 2 → 64 ReLU → 784 sigmoid (Bernoulli pixel probabilities).
- Objective: summed pixel binary cross entropy plus KL(q(z|x) || N(0,I)), β = 1.
- Adam, learning rate 0.001, batch size 256, seed 147; 48 epochs on CPU.
- 25 saved stages: random initialization and every second epoch through epoch 48.
- Held-out mean-code reconstruction MSE: **0.23248 → 0.03180**. This metric is
  evaluated on 1,500 held-out images and is distinct from the stochastic ELBO.
- Learned decoder weights are exported in little-endian float16. Across all
  saved stages, the largest pixel-probability change from export was **0.001236**
  on held-out encoder means and prior draws. Inference accumulates in JavaScript
  numbers and writes float32 pixel probabilities before canvas display.

`decoders.bin` contains 25 consecutive blocks of 51,152 float16 values:
64×2 hidden weights, 64 hidden biases, 784×64 output weights, and 784 output
biases. Matrices are output-major, for contiguous dot products in JavaScript.
The binary is 2,557,600 bytes; `model.json` contains the run metadata, checksums,
held-out images (base64 uint8), and per-stage encoder outputs. No optimizer,
training set, tensor framework, or inference service is shipped to students.

The training button **replays saved checkpoints**; it does not train online.
The stage slider, pause button and trained-model shortcut all select actual
saved weights. Dragging any latent location computes a fresh forward pass,
including locations that were never among the encoded examples. Output images
show decoder means; they are not Bernoulli pixel samples. Reconstructions start
at μ(x), so the replay has a stable reference image. Some locations, especially
away from the encoded examples, produce ambiguous or poorly formed digits.

## Reproduce

Only Python and NumPy are needed. Download `mnist.npz` from the source below
outside the repository, then run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python scripts/train_mnist_vae.py --gradient-check
OPENBLAS_NUM_THREADS=1 python scripts/train_mnist_vae.py \
  --data /tmp/vae-mnist.npz --out book/_static/vae-mnist
```

The optional `--reference /tmp/vae-final-reference.npz` exports the full-precision
network for development verification. It is not a runtime dependency and should
not be committed. The gradient check compares the manually implemented backward
pass against central finite differences. Export also checks float16 forward-pass
agreement at every saved stage. Training and exporting took approximately three
minutes on the development CPU; execution time depends on the BLAS library.

## Dataset attribution and license

MNIST is by **Yann LeCun and Corinna Cortes**, derived from NIST handwritten
digits. Embedded images retain **Creative Commons Attribution-Share Alike 3.0**;
that attribution and license apply to the embedded image examples independently
of the repository's CC BY 4.0 license. Learned model weights and the new widget
code are provided under this repository's license.

- Source download: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
- Dataset documentation and license: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
- Image license: https://creativecommons.org/licenses/by-sa/3.0/
- Original MNIST homepage: http://yann.lecun.com/exdb/mnist/

The dataset checksum and decoder checksum are recorded in `model.json`.
