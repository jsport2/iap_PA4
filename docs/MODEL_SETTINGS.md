# Model training and naming

All 100 image classifiers were created with the Google Teachable Machine Image
Project:

<https://teachablemachine.withgoogle.com/train/image>

## Training settings

| Setting | Value |
|---|---:|
| Epochs | 50 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Input size | 224 x 224 RGB |
| Output order | airplane, monkey, truck |
| Export format | TensorFlow/Keras |

Teachable Machine did not expose a model-training seed. Dataset selection is
deterministic: runs 1--3 use seed `20261114`, and runs 4--10 use seed base
`20261115`.

## Model layout

Each condition contains ten downloaded models with uniform run names:

```text
models/<condition>/run_01/keras_model.h5
models/<condition>/run_01/labels.txt
...
models/<condition>/run_10/keras_model.h5
models/<condition>/run_10/labels.txt
```

The ten condition names are `model_clean`, `model_mixed_p10`,
`model_mixed_p20`, `model_mixed_p30`, `model_systematic_p10`,
`model_systematic_p20`, `model_systematic_p30`, `model_unseen_p10`,
`model_unseen_p20`, and `model_unseen_p30`.
