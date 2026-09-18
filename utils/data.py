"""
Fixed 200-sample Flickr30k evaluation set for SmolVLM.

The first run reads the already-cached Flickr30k dataset and creates
a small local evaluation set.

All future experiments use exactly the same 200 samples.
"""

from pathlib import Path
import json

from datasets import load_dataset
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
IMAGE_DIR = EVAL_DIR / "images"
METADATA_FILE = EVAL_DIR / "metadata.json"

N_SAMPLES = 200


def prepare_evaluation_set(n_samples=N_SAMPLES):

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # If already prepared, don't download/load the dataset again
    if METADATA_FILE.exists():

        print("[cached] Loading local evaluation set...")

        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        dataset = []

        for item in metadata:

            image_path = PROJECT_ROOT / item["image_path"]

            image = Image.open(image_path).convert("RGB")

            dataset.append({
                "image": image,
                "caption": item["caption"],
                "img_id": item["img_id"],
                "filename": item["filename"],
            })

        print(f"Loaded {len(dataset)} local evaluation samples.")

        return dataset

    print("Creating local evaluation set from cached Flickr30k...")

    # This should use the dataset already downloaded to HF cache.
    ds = load_dataset(
        "nlphuji/flickr30k",
        split="test"
    )

    n = min(n_samples, len(ds))

    metadata = []
    dataset = []

    for i in range(n):

        sample = ds[i]

        image = sample["image"].convert("RGB")

        filename = sample["filename"]

        # Avoid filename collisions
        output_filename = f"{i:04d}_{filename}"

        output_path = IMAGE_DIR / output_filename

        image.save(output_path, format="JPEG", quality=95)

        item = {
            "image_path": str(
                output_path.relative_to(PROJECT_ROOT)
            ),
            "caption": sample["caption"],
            "img_id": str(sample["img_id"]),
            "filename": filename,
        }

        metadata.append(item)

        dataset.append({
            "image": image,
            "caption": sample["caption"],
            "img_id": str(sample["img_id"]),
            "filename": filename,
        })

        if (i + 1) % 25 == 0:
            print(f"Prepared {i + 1}/{n} images")

    # Save metadata
    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(f"Created local evaluation set: {n} images")
    print(f"Images: {IMAGE_DIR}")
    print(f"Metadata: {METADATA_FILE}")

    return dataset


def get_caption_subset(
    n_samples=200,
    split="test"
):
    """
    Main function used by the model scripts.
    """

    if split != "test":
        raise ValueError(
            "This evaluation loader uses the Flickr30k test split."
        )

    return prepare_evaluation_set(n_samples)


if __name__ == "__main__":

    dataset = get_caption_subset(n_samples=200)

    print()
    print("Number of samples:", len(dataset))

    print("First sample keys:", dataset[0].keys())

    print("Image size:", dataset[0]["image"].size)

    print("Reference captions:")

    for caption in dataset[0]["caption"]:
        print("-", caption)