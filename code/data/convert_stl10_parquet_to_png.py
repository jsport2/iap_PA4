"""Extract the labeled STL-10 Parquet mirror into class-organized PNG files."""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
from collections import Counter
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA_ROOT = REPOSITORY_ROOT / "datasets" / "source" / "STL10"

import pyarrow.parquet as pq
from PIL import Image


OFFICIAL_CLASS_NAMES = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)

# The source dataset used string labels "1" through "10". During the
# Hugging Face Parquet conversion, those strings were sorted lexicographically
# as 1, 10, 2, 3, ..., 9 before being encoded as integers 0 through 9.
PARQUET_LABEL_TO_CLASS = (
    "airplane",  # source label "1"
    "truck",     # source label "10"
    "bird",      # source label "2"
    "car",       # source label "3"
    "cat",       # source label "4"
    "deer",      # source label "5"
    "dog",       # source label "6"
    "horse",     # source label "7"
    "monkey",    # source label "8"
    "ship",      # source label "9"
)

RAW_DIR = SOURCE_DATA_ROOT / "raw" / "huggingface_parquet"
OUTPUT_DIR = SOURCE_DATA_ROOT / "png"
MANIFEST_PATH = SOURCE_DATA_ROOT / "conversion_manifest.csv"

SPLITS = {
    "train": {
        "path": RAW_DIR / "train.parquet",
        "expected_per_class": 500,
    },
    "test": {
        "path": RAW_DIR / "test.parquet",
        "expected_per_class": 800,
    },
}


def validate_png(image_bytes: bytes, split: str, row_index: int) -> None:
    """Validate the embedded image without altering its encoded bytes."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.format != "PNG":
            raise ValueError(f"{split} row {row_index}: expected PNG, got {image.format}")
        if image.size != (96, 96):
            raise ValueError(f"{split} row {row_index}: expected 96x96, got {image.size}")
        if image.mode != "RGB":
            raise ValueError(f"{split} row {row_index}: expected RGB, got {image.mode}")
        image.verify()


def extract_split(
    split: str,
    parquet_path: Path,
    manifest_writer: csv.writer,
) -> Counter[str]:
    """Extract one Parquet split and return class counts."""
    counters: Counter[str] = Counter()
    row_index = 0

    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=256, columns=["image", "label"]):
        images = batch.column("image").to_pylist()
        labels = batch.column("label").to_pylist()

        for image_record, label in zip(images, labels, strict=True):
            label = int(label)
            if not 0 <= label < len(PARQUET_LABEL_TO_CLASS):
                raise ValueError(f"{split} row {row_index}: invalid label {label}")

            class_name = PARQUET_LABEL_TO_CLASS[label]
            official_label = OFFICIAL_CLASS_NAMES.index(class_name)
            image_bytes = image_record["bytes"]
            source_path = image_record.get("path") or ""
            if not image_bytes:
                raise ValueError(f"{split} row {row_index}: missing image bytes")

            validate_png(image_bytes, split, row_index)

            counters[class_name] += 1
            output_name = f"{counters[class_name]}.png"
            relative_output = Path("png") / split / class_name / output_name
            output_path = SOURCE_DATA_ROOT / relative_output
            output_path.write_bytes(image_bytes)

            manifest_writer.writerow(
                (
                    split,
                    row_index,
                    label,
                    official_label,
                    class_name,
                    source_path,
                    relative_output.as_posix(),
                    hashlib.sha256(image_bytes).hexdigest(),
                )
            )
            row_index += 1

    return counters


def main() -> None:
    for config in SPLITS.values():
        parquet_path = config["path"]
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing raw file: {parquet_path}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in SPLITS:
        for class_name in OFFICIAL_CLASS_NAMES:
            (OUTPUT_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

    results: dict[str, Counter[str]] = {}
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            (
                "split",
                "source_row",
                "parquet_label_id",
                "official_label_id",
                "class_name",
                "source_path",
                "output_path",
                "sha256",
            )
        )

        for split, config in SPLITS.items():
            counts = extract_split(split, config["path"], writer)
            expected = config["expected_per_class"]
            for class_name in OFFICIAL_CLASS_NAMES:
                actual = counts[class_name]
                if actual != expected:
                    raise ValueError(
                        f"{split}/{class_name}: expected {expected} images, found {actual}"
                    )
            results[split] = counts

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    for split, counts in results.items():
        print(f"{split}: total={sum(counts.values())}, per_class={dict(counts)}")


if __name__ == "__main__":
    main()
