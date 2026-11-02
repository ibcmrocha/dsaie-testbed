"""Train/export the MNIST latent-space widget. Requires only NumPy.

OPENBLAS_NUM_THREADS=1 python scripts/train_mnist_vae.py --gradient-check
OPENBLAS_NUM_THREADS=1 python scripts/train_mnist_vae.py \
    --data /tmp/vae-mnist.npz --out book/_static/vae-mnist

MNIST source: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
Only the official training split is optimized; displayed examples are held out.
"""
import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np

DIGITS = [0, 1, 8]
HIDDEN = 64


def initialize(rng):
    def weight(a, b, scale):
        return (rng.normal(size=(a, b)) * scale).astype(np.float32)
    return {
        "enc_w": weight(784, HIDDEN, np.sqrt(2 / 784)),
        "enc_b": np.zeros(HIDDEN, np.float32),
        "mu_w": weight(HIDDEN, 2, 0.1 / np.sqrt(HIDDEN)),
        "mu_b": np.zeros(2, np.float32),
        "lv_w": weight(HIDDEN, 2, 0.1 / np.sqrt(HIDDEN)),
        "lv_b": np.zeros(2, np.float32),
        "dec_w": weight(2, HIDDEN, 1),
        "dec_b": np.zeros(HIDDEN, np.float32),
        "out_w": weight(HIDDEN, 784, 1 / np.sqrt(HIDDEN)),
        "out_b": np.zeros(784, np.float32),
    }


def encode(p, x):
    h = np.maximum(x @ p["enc_w"] + p["enc_b"], 0)
    return h @ p["mu_w"] + p["mu_b"], h @ p["lv_w"] + p["lv_b"]


def decode(p, z):
    h = np.maximum(z @ p["dec_w"] + p["dec_b"], 0)
    logits = h @ p["out_w"] + p["out_b"]
    return 1 / (1 + np.exp(-np.clip(logits, -80, 80)))


def objective(p, x, eps):
    n = len(x)
    h = np.maximum(x @ p["enc_w"] + p["enc_b"], 0)
    mu, lv = h @ p["mu_w"] + p["mu_b"], h @ p["lv_w"] + p["lv_b"]
    std = np.exp(0.5 * lv)
    z = mu + std * eps
    dh = np.maximum(z @ p["dec_w"] + p["dec_b"], 0)
    logits = dh @ p["out_w"] + p["out_b"]
    y = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    recon = np.sum(np.logaddexp(0, logits) - x * logits) / n
    kl = 0.5 * np.sum(mu * mu + np.exp(lv) - lv - 1) / n
    d = (y - x) / n
    grads = {"out_w": dh.T @ d, "out_b": d.sum(axis=0)}
    d = (d @ p["out_w"].T) * (dh > 0)
    grads.update(dec_w=z.T @ d, dec_b=d.sum(axis=0))
    dz = d @ p["dec_w"].T
    dm = dz + mu / n
    dl = dz * eps * std * 0.5 + (np.exp(lv) - 1) * (0.5 / n)
    grads.update(mu_w=h.T @ dm, mu_b=dm.sum(axis=0),
                 lv_w=h.T @ dl, lv_b=dl.sum(axis=0))
    d = (dm @ p["mu_w"].T + dl @ p["lv_w"].T) * (h > 0)
    grads.update(enc_w=x.T @ d, enc_b=d.sum(axis=0))
    return float(recon + kl), grads


def gradient_check():
    rng = np.random.default_rng(46)
    p = {k: v.astype(np.float64) for k, v in initialize(rng).items()}
    x, eps = rng.random((3, 784)), rng.normal(size=(3, 2))
    _, grads = objective(p, x, eps)
    worst = 0
    for name, values in p.items():
        for _ in range(3):
            index = tuple(rng.integers(s) for s in values.shape)
            original, step = values[index], 1e-5
            values[index] = original + step
            plus, _ = objective(p, x, eps)
            values[index] = original - step
            minus, _ = objective(p, x, eps)
            values[index] = original
            numerical = (plus - minus) / (2 * step)
            analytic = grads[name][index]
            error = abs(numerical - analytic) / max(1, abs(numerical), abs(analytic))
            worst = max(worst, error)
            assert error < 1e-5, (name, index, numerical, analytic)
    print(f"Gradient check passed: worst relative error {worst:.3g}")


def train(args):
    rng = np.random.default_rng(args.seed)
    with np.load(args.data) as source:
        labels = source["y_train"]
        selected = np.isin(labels, DIGITS)
        x = source["x_train"][selected].reshape(-1, 784).astype(np.float32) / 255
        # Thirty held-out examples of each digit, interleaved by class.
        test_labels = source["y_test"]
        groups = [np.flatnonzero(test_labels == digit)[:30] for digit in DIGITS]
        indices = np.array(groups).T.reshape(-1)
        images = source["x_test"][indices].reshape(-1, 784)
        sample_labels = test_labels[indices].tolist()
        validation = source["x_test"][np.isin(test_labels, DIGITS)][:1500].reshape(-1, 784).astype(np.float32) / 255
    examples = images.astype(np.float32) / 255
    p = initialize(rng)
    moments = {k: np.zeros_like(v) for k, v in p.items()}
    velocity = {k: np.zeros_like(v) for k, v in p.items()}
    steps, history, weights = 0, [], []
    started = time.perf_counter()
    checkpoint_errors = []

    def checkpoint(epoch):
        mu, lv = encode(p, examples)
        vm, vl = encode(p, validation)
        mse = float(np.mean((decode(p, vm) - validation) ** 2))
        kl = float(0.5 * np.mean(np.sum(vm ** 2 + np.exp(vl) - vl - 1, axis=1)))
        # Output-major layout: contiguous dot products for browser inference.
        arrays = [p["dec_w"].T, p["dec_b"], p["out_w"].T, p["out_b"]]
        exported = np.concatenate([a.reshape(-1) for a in arrays]).astype("<f2")
        restored = exported.astype(np.float32)
        split = np.cumsum([a.size for a in arrays])[:-1]
        a, b, c, d = np.split(restored, split)
        q = {"dec_w": a.reshape(HIDDEN, 2).T, "dec_b": b,
             "out_w": c.reshape(784, HIDDEN).T, "out_b": d}
        probes = np.concatenate([mu, rng.normal(size=(32, 2)).astype(np.float32)])
        error = float(np.max(np.abs(decode(p, probes) - decode(q, probes))))
        assert error < 0.006, f"Float16 export changed a pixel by {error}"
        checkpoint_errors.append(error)
        weights.append(exported.tobytes())
        history.append({"epoch": epoch, "mse": round(mse, 6), "kl": round(kl, 5),
                        "mu": np.round(mu.astype(np.float64), 5).tolist(),
                        "sigma": np.round(np.exp(0.5 * lv).astype(np.float64), 5).tolist()})
        print(f"epoch {epoch:2d} | held-out MSE {mse:.5f} | KL {kl:.3f} | elapsed {time.perf_counter()-started:.1f}s", flush=True)

    checkpoint(0)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x))
        for first in range(0, len(x), args.batch):
            batch = x[order[first:first + args.batch]]
            eps = rng.normal(size=(len(batch), 2)).astype(np.float32)
            loss, grads = objective(p, batch, eps)
            assert np.isfinite(loss)
            steps += 1
            for key, grad in grads.items():
                moments[key] *= 0.9
                moments[key] += 0.1 * grad
                velocity[key] *= 0.999
                velocity[key] += 0.001 * grad ** 2
                p[key] -= args.learning_rate * (moments[key] / (1 - 0.9 ** steps)) / (np.sqrt(velocity[key] / (1 - 0.999 ** steps)) + 1e-8)
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            checkpoint(epoch)
    args.out.mkdir(parents=True, exist_ok=True)
    binary = b"".join(weights)
    (args.out / "decoders.bin").write_bytes(binary)
    metadata = {
        "version": 1, "digits": DIGITS, "hidden": HIDDEN, "latent": 2,
        "imageSize": 28, "weightCount": len(binary) // len(weights) // 2,
        "format": "little-endian-float16", "weightOrder": ["decoder_hidden_weights_output_major", "decoder_hidden_bias", "decoder_output_weights_output_major", "decoder_output_bias"],
        "training": {"seed": args.seed, "epochs": args.epochs, "batchSize": args.batch,
                     "learningRate": args.learning_rate, "examples": len(x),
                     "objective": "Bernoulli negative log likelihood + KL(q(z|x) || N(0,I)); beta=1",
                     "encoder": [784, HIDDEN, "mean:2, logVariance:2"],
                     "decoder": [2, HIDDEN, 784], "hiddenActivation": "ReLU",
                     "outputActivation": "sigmoid", "elapsedSeconds": round(time.perf_counter() - started, 2)},
        "labels": sample_labels, "images": base64.b64encode(images.tobytes()).decode(),
        "checkpoints": history, "exportMaxPixelError": max(checkpoint_errors),
        "weightsSha256": hashlib.sha256(binary).hexdigest(),
        "datasetSha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "source": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
    }
    (args.out / "model.json").write_text(json.dumps(metadata, separators=(",", ":")) + "\n")
    # Development checks only, not an additional runtime model asset.
    if args.reference:
        np.savez(args.reference, **p)
    print(f"Exported {len(weights)} decoders, {len(binary):,} bytes; maximum float16 pixel error {max(checkpoint_errors):.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-check", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path, default=Path("book/_static/vae-mnist"))
    parser.add_argument("--reference", type=Path, help="Optional full-precision development reference; do not publish")
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--checkpoint-every", type=int, default=2)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=147)
    arguments = parser.parse_args()
    if arguments.gradient_check:
        gradient_check()
    else:
        if not arguments.data:
            parser.error("--data must point to the downloaded MNIST npz")
        train(arguments)
