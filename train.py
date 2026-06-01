import ctypes, gzip, os, struct, urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
BATCH, IN, K, C, CONV, POOL, FLAT = 64, 28, 3, 8, 26, 13, 1352
libname = "conv.dll" if os.name == "nt" else "libconv2d.so"
LIB = ctypes.CDLL(str(ROOT / libname))
PTR = ctypes.POINTER(ctypes.c_float)
LIB.conv2d_forward.argtypes = [PTR, PTR, PTR, PTR, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def load_mnist(split):
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {"train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
             "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")}[split]
    data = ROOT / "data"; data.mkdir(exist_ok=True)
    for f in files:
        if not (data / f).exists():
            urllib.request.urlretrieve(base + f, data / f)
    with gzip.open(data / files[0], "rb") as f:
        _, n, h, w = struct.unpack(">IIII", f.read(16))
        x = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 1, h, w).astype("float32") / 255.0
    with gzip.open(data / files[1], "rb") as f:
        f.read(8); y = np.frombuffer(f.read(), dtype=np.uint8).astype("int64")
    return torch.from_numpy(x), torch.from_numpy(y)


class CConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernels, bias):
        x, kernels, bias = x.contiguous(), kernels.contiguous(), bias.contiguous()
        out = torch.empty((x.shape[0], C, CONV, CONV), dtype=torch.float32)
        LIB.conv2d_forward(x.detach().numpy().ctypes.data_as(PTR),
                           kernels.detach().numpy().ctypes.data_as(PTR),
                           bias.detach().numpy().ctypes.data_as(PTR),
                           out.numpy().ctypes.data_as(PTR),
                           x.shape[0], IN, IN, C, K, K)
        ctx.save_for_backward(x, kernels)
        return out

    @staticmethod
    def backward(ctx, g):
        x, kernels = ctx.saved_tensors
        gx, gk = torch.zeros_like(x), torch.zeros_like(kernels)
        gb = g.sum((0, 2, 3))
        for y in range(K):
            for z in range(K):
                xs = x[:, 0, y:y + CONV, z:z + CONV]
                gk[:, y, z] = (g * xs[:, None]).sum((0, 2, 3))
                gx[:, 0, y:y + CONV, z:z + CONV] += (g * kernels[:, y, z].view(1, C, 1, 1)).sum(1)
        return gx, gk, gb


def logits(x, kernels, cbias, w, b):
    z = F.max_pool2d(F.relu(CConv.apply(x, kernels, cbias)), 2).flatten(1)
    return z @ w.t() + b


def accuracy(loader, kernels, cbias, w, b):
    good = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = F.softmax(logits(x, kernels, cbias, w, b), 1).argmax(1)
            good += (pred == y).sum().item(); total += len(y)
    return good / total


def plot(values, path, ylabel):
    plt.figure()
    for name, vals in values:
        plt.plot(range(1, len(vals) + 1), vals, marker="o", label=name)
    plt.xlabel("epoch"); plt.ylabel(ylabel); plt.legend(); plt.tight_layout()
    plt.savefig(ROOT / path); plt.close()


def main():
    torch.manual_seed(1)
    xtr, ytr = load_mnist("train"); xte, yte = load_mnist("test")
    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(TensorDataset(xte, yte), batch_size=256)
    kernels = torch.nn.Parameter(torch.randn(C, K, K) * 0.1)
    cbias = torch.nn.Parameter(torch.zeros(C))
    w = torch.nn.Parameter(torch.randn(10, FLAT) * 0.05)
    b = torch.nn.Parameter(torch.zeros(10))
    opt = torch.optim.Adam([kernels, cbias, w, b], lr=0.001)
    log, losses, train_accs, test_accs = [], [], [], []

    for ep in range(1, 6):
        total, good, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            out = logits(x, kernels, cbias, w, b)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y); good += (out.argmax(1) == y).sum().item(); total += len(y)
        train_acc = good / total; test_acc = accuracy(test_loader, kernels, cbias, w, b)
        epoch_loss = loss_sum / total
        losses.append(epoch_loss); train_accs.append(train_acc); test_accs.append(test_acc)
        line = f"Epoch {ep} Loss {epoch_loss:.4f} Train Accuracy {train_acc:.4f} Test Accuracy {test_acc:.4f}"
        print(line); log.append(line)

    torch.save({"kernels": kernels.detach(), "conv_bias": cbias.detach(), "linear_w": w.detach(), "linear_b": b.detach()}, ROOT / "model_weights.pth")
    (ROOT / "training_log.txt").write_text("\n".join(log) + "\n")
    plot([("loss", losses)], "loss_plot.png", "loss")
    plot([("train", train_accs), ("test", test_accs)], "accuracy_plot.png", "accuracy")

    with torch.no_grad():
        pred = F.softmax(logits(xte[:10], kernels, cbias, w, b), 1).argmax(1)
    with open(ROOT / "samples.bin", "wb") as f, open(ROOT / "python_predictions.txt", "w") as txt:
        for i in range(10):
            print(f"sample index {i} python prediction {pred[i].item()} true label {yte[i].item()}")
            txt.write(f"Sample {i}\nPython Prediction: {pred[i].item()}\nTrue Label: {yte[i].item()}\n\n")
            f.write(struct.pack("iii", i, int(pred[i]), int(yte[i])))
            f.write(xte[i].contiguous().numpy().astype("float32").tobytes())

    shown = F.softmax(logits(xte[:5], kernels, cbias, w, b), 1).argmax(1)
    plt.figure(figsize=(8, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1); plt.imshow(xte[i, 0], cmap="gray")
        plt.title(f"Pred:{shown[i].item()} True:{yte[i].item()}"); plt.axis("off")
    plt.tight_layout(); plt.show(block=False); plt.pause(3); plt.close()


if __name__ == "__main__":
    main()
