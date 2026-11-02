"""Small supervised MLP. No pretrained models, external images, or API calls."""
from pathlib import Path
import json
import numpy as np

LABELS = ("RED", "GREEN", "BLUE")
COLORS = np.array([[.85, .13, .10], [.13, .78, .19], [.12, .26, .88]])


def features(rgb):
    rgb = np.asarray(rgb, dtype=float)
    return rgb / np.maximum(rgb.sum(axis=-1, keepdims=True), 1e-8)


def make_data(seed, n):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    rgb = COLORS[y] * rng.uniform(.35, 1.15, (n, 1))
    rgb *= rng.uniform(.8, 1.2, (n, 3))
    rgb += rng.uniform(0, .12, (n, 1)) + rng.normal(0, .035, (n, 3))
    return features(np.clip(rgb, 0, 1)), y


def probabilities(x, w):
    h = np.tanh(x @ w['w1'] + w['b1'])
    z = h @ w['w2'] + w['b2']
    z -= np.max(z, axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def train(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x, y = make_data(2026, 6000)
    xt, yt = make_data(2027, 1500)
    rng = np.random.default_rng(7)
    w = {'w1': rng.normal(0, .5, (3, 16)), 'b1': np.zeros(16),
         'w2': rng.normal(0, .3, (16, 3)), 'b2': np.zeros(3)}
    history = []
    for epoch in range(500):
        h = np.tanh(x @ w['w1'] + w['b1'])
        p = probabilities(x, w)
        if epoch % 100 == 0 or epoch == 499:
            history.append({'epoch': epoch, 'cross_entropy': float(-np.log(p[np.arange(len(y)), y]).mean())})
        dz = p.copy()
        dz[np.arange(len(y)), y] -= 1
        dz /= len(y)
        dh = (dz @ w['w2'].T) * (1-h*h)
        gradients = {'w2': h.T @ dz, 'b2': dz.sum(0), 'w1': x.T @ dh, 'b1': dh.sum(0)}
        for name in w:
            w[name] -= .45 * gradients[name]
    pred = probabilities(xt, w).argmax(1)
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (yt, pred), 1)
    result = {'train_seed': 2026, 'test_seed': 2027, 'initialization_seed': 7,
              'train_n': 6000, 'test_n': 1500, 'epochs': 500,
              'test_accuracy': float(np.mean(pred == yt)), 'confusion_matrix': cm.tolist(),
              'history': history, 'architecture': '3 normalized RGB -> tanh(16) -> softmax(3)',
              'note': 'Synthetic test only; not evidence of general real-world recognition.'}
    np.savez(output / 'color_mlp.npz', **w)
    (output / 'training.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


class Classifier:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as f:
            self.weights = {k: f[k] for k in f.files}

    def predict(self, image):
        """Fixed known cargo ROI. This is classification, not object detection."""
        height, width = image.shape[:2]
        roi = image[height//2-10:height//2+10, width//2-10:width//2+10]
        rgb = np.median(roi.reshape(-1, 3), axis=0) / 255
        p = probabilities(features(rgb)[None, :], self.weights)[0]
        saturation = float((rgb.max()-rgb.min()) / max(rgb.max(), 1e-8))
        accepted = bool(p.max() >= .80 and saturation >= .30 and rgb.max() >= .08)
        return {'label': int(p.argmax()) if accepted else None,
                'confidence': float(p.max()), 'rgb': rgb.tolist(), 'saturation': saturation,
                'accepted': accepted}


if __name__ == '__main__':
    print(json.dumps(train(Path('models')), indent=2))
