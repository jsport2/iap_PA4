# Exported models

This folder contains 100 Teachable Machine Keras exports: ten data conditions
times ten training runs. Every condition uses the same structure:

```text
models/<condition>/run_01/keras_model.h5
models/<condition>/run_01/labels.txt
...
models/<condition>/run_10/keras_model.h5
models/<condition>/run_10/labels.txt
```

Every `labels.txt` file uses the output order `airplane`, `monkey`, `truck`.
Training settings are documented in `docs/MODEL_SETTINGS.md`.
