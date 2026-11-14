# Exported models

There are 100 Keras exports here: ten training conditions with ten independent
training runs per condition. All models were created in Google Teachable
Machine and follow the same file layout:

```text
models/<condition>/run_01/keras_model.h5
models/<condition>/run_01/labels.txt
...
models/<condition>/run_10/keras_model.h5
models/<condition>/run_10/labels.txt
```

Every `labels.txt` file lists the outputs in the order `airplane`, `monkey`,
`truck`. The training parameters and condition names are recorded in
`docs/MODEL_SETTINGS.md`.
