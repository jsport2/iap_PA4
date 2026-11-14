# Model training and naming

The 100 classifiers were trained with the Image Project on
[Google Teachable Machine](https://teachablemachine.withgoogle.com/train/image).
Each model was exported in TensorFlow/Keras format.

## Training settings

| Setting | Value |
|---|---:|
| Epochs | 50 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Input size | 224 × 224 RGB |
| Output order | airplane, monkey, truck |
| Export format | TensorFlow/Keras |

Teachable Machine does not expose a seed for model training, so its internal
training randomness could not be fixed. Dataset selection is deterministic:
runs 1--3 use seed `20261114`, while runs 4--10 use the base seed `20261115`.
The ten exported models per condition capture the remaining variation between
training runs.

## File names

Each condition follows the same layout:

```text
models/<condition>/run_01/keras_model.h5
models/<condition>/run_01/labels.txt
...
models/<condition>/run_10/keras_model.h5
models/<condition>/run_10/labels.txt
```

The ten condition names are:

- `model_clean`
- `model_mixed_p10`, `model_mixed_p20`, `model_mixed_p30`
- `model_systematic_p10`, `model_systematic_p20`, `model_systematic_p30`
- `model_unseen_p10`, `model_unseen_p20`, `model_unseen_p30`
