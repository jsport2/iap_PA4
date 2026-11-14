"""Smoke-test a downloaded Teachable Machine Keras model on a fixed test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("test_model_dir", type=Path)
    return parser.parse_args()


def read_labels(path: Path) -> list[str]:
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid label line: {line!r}")
        labels.append(parts[1])
    return labels


def main() -> None:
    args = parse_args()
    model_path = args.model_dir / "keras_model.h5"
    labels_path = args.model_dir / "labels.txt"

    model = load_model(model_path, compile=False)
    class_names = read_labels(labels_path)

    if len(model.outputs) != 1 or model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"Output shape {model.output_shape} does not match {len(class_names)} labels"
        )

    input_height = int(model.input_shape[1])
    input_width = int(model.input_shape[2])
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)

    for true_index, true_label in enumerate(class_names):
        test_dir = args.test_model_dir / true_label / "test"
        image_paths = sorted(test_dir.glob("*.png"), key=lambda path: int(path.stem))
        if not image_paths:
            raise ValueError(f"No test images found in {test_dir}")

        for image_path in image_paths:
            encoded = np.frombuffer(image_path.read_bytes(), dtype=np.uint8)
            image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Cannot read image: {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.resize(
                image_rgb, (input_width, input_height), interpolation=cv2.INTER_AREA
            )
            image = image_rgb.astype(np.float32).reshape(
                1, input_height, input_width, 3
            )
            image = (image / 127.5) - 1.0

            prediction = model.predict(image, verbose=0)[0]
            predicted_index = int(np.argmax(prediction))
            confusion[true_index, predicted_index] += 1

    total = int(confusion.sum())
    correct = int(np.trace(confusion))

    print(f"Model: {model_path}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Labels: {class_names}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion)
    print("Per-class accuracy:")
    for index, class_name in enumerate(class_names):
        class_total = int(confusion[index].sum())
        class_correct = int(confusion[index, index])
        accuracy = class_correct / class_total * 100
        print(f"  {class_name}: {accuracy:.2f}% ({class_correct}/{class_total})")
    print(f"Overall accuracy: {correct / total * 100:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
