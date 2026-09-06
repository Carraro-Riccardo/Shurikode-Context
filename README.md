# Shurikode Context

Shurikode Context is a context-aware extension of [Riccardo Toniolo's original Shurikode project](https://github.com/RiccardoTonioloDev/Shurikode), a highly redundant two-dimensional code that represents integers from `0` to `255`.

This version keeps the original encoding unchanged and replaces the decoder with retrained models designed for a less constrained input: the Shurikode may be degraded and may occupy only part of the image, with a visible border or surrounding background. It also introduces a **no Shurikode** class, so the decoder can report that the input does not contain a recognizable code.

## Differences from the original project

| | Original Shurikode | Shurikode Context |
|---|---|---|
| Encoded values | `0`–`255` | `0`–`255` (unchanged) |
| Decoder classes | 256 | 257 |
| Negative input | Always classified as one of `0`–`255` | Label `256` means **no Shurikode** |
| Decoder result | Integer label | `(label, confidence)` tuple |
| Context emphasis | Synthetic degradation included random padding, plus limited real-data fine-tuning | Retrained specifically for degraded codes with more surrounding border/background |
| Inference preprocessing | Resize to 400×400 | RGB conversion, resize to 224×224, ImageNet normalization |
| Python package | `shurikode` | `shurikodecontext` |
| Decoder backbones | ResNet-18, ResNet-34, ResNet-50 | Retrained ResNet-18, ResNet-34, ResNet-50 |

The original training pipeline already included several synthetic degradations, including random padding, and a real-image fine-tuning path. The two defining changes here are therefore the explicit negative class and the stronger emphasis on non-tightly-cropped contextual inputs, rather than the first-ever use of padding.

The context-aware decoder classifies the complete image supplied by the caller; it does not run a separate object detector or crop the code first. The code must therefore remain visible enough after the whole image is resized to 224×224. The returned confidence is the maximum softmax score, not a calibrated probability.

## Encoding format

<img src="./docs_images/Shurikode_encoding.png" alt="Shurikode encoding strategy" width="300">

The 8-bit representation of a value is repeated four times across three levels:

- **External level** (red and orange): the bits are placed on every side, following a clockwise rotation.
- **Middle level** (blue and purple): the bits follow the same rotation, with a one-position shift; this level is inverted.
- **Internal level** (light and dark green): the bits continue clockwise across the innermost three lines.

The corner squares are parity bits for the external level. The encoder also adds a white border around the code.

The redundancy and rotational symmetry are intended to preserve decodability under strong visual degradation. Shurikode Context extends this goal to images in which the degraded code is not perfectly cropped.

## Installation

Clone this repository and install it with pip:

```bash
git clone https://github.com/Carraro-Riccardo/Shurikode-Context.git
cd Shurikode-Context
pip install .
```

The three model checkpoints are included in the installed package, so no separate model download is required.

## Usage

### Encode a value

```python
from shurikodecontext import Enc

encoder = Enc(size=10)
encoded = encoder.encode(255)

image = encoded.get_PIL_image()
encoded.save("shurikode-255.png")
```

`size` controls the scale of the generated image. Values must be between `0` and `255`, inclusive.

### Decode an image

```python
from PIL import Image
from shurikodecontext import Dec

decoder = Dec("L")

with Image.open("input.png") as image:
    label, confidence = decoder(image)

if label == 256:
    print(f"No Shurikode detected (confidence: {confidence:.3f})")
else:
    print(f"Decoded value: {label} (confidence: {confidence:.3f})")
```

Available decoder sizes are:

| Size | Backbone | Relative cost |
|---|---|---|
| `"S"` | ResNet-18 | Lowest |
| `"M"` | ResNet-34 | Intermediate |
| `"L"` | ResNet-50 | Highest |

The decoder accepts either:

- a Pillow image, which is converted to RGB automatically; or
- a PyTorch tensor in `CHW` format, optionally with a single-image batch dimension (`1CHW`), with values expected in the `[0, 1]` range.

No manual resize or normalization is required.

## Model checkpoints

The runtime loads its weights from `src/shurikodecontext/ml_model/`. These files must remain in the source distribution for `pip install .` and in the resulting wheel.

`src/shurikodecontext/ml_model/` is the canonical location for the released checkpoints. Files placed under `ml/checkpoints/` are training-side artifacts: they are not imported by the runtime, selected by `setup.py` or `MANIFEST.in`, or included in the wheel. Any duplicate copies there can be removed without affecting installation or inference.

## Attribution

The Shurikode format, encoder, and original decoder are based on [Riccardo Toniolo's Shurikode](https://github.com/RiccardoTonioloDev/Shurikode). This repository changes the package namespace, inference contract, preprocessing, class set, training data, and model checkpoints to support contextual inputs and negative examples.

Released under the MIT License. See [LICENSE](./LICENSE).
