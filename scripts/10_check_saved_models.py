import torch
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

files = [
    "quantized_model.pt",
    "pruned_model.pt",
    "optimized_model.pt",
]

for filename in files:
    path = RESULTS_DIR / filename

    print("\n" + "=" * 60)
    print(filename)

    if not path.exists():
        print("NOT FOUND")
        continue

    print(f"File size: {path.stat().st_size / (1024 * 1024):.2f} MB")
    print("Loading...")

    try:
        obj = torch.load(path, weights_only=False)

        print("Loaded successfully")
        print("Python type:", type(obj))

        if isinstance(obj, dict):
            print("Dictionary keys:")
            for key in list(obj.keys())[:10]:
                print("  ", key)

            print("Total keys:", len(obj))

        elif hasattr(obj, "state_dict"):
            print("This is a complete PyTorch model object")
            print("Model class:", type(obj).__name__)

        else:
            print("Unknown saved object type")

    except Exception as e:
        print("ERROR:", e)