"""Print the class names inside a weights file, so you can match them in config.yaml.

    python tools\\inspect_model.py models\\mask.pt
If the names differ from config.yaml (e.g. 'NO-Mask' instead of 'no_mask'), either
edit `classes:` in config.yaml or add a `rename:` map for that model.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from common.models import load_model  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python tools\\inspect_model.py <weights.pt>")
    m = load_model(sys.argv[1], coco=False)
    if m is None:
        sys.exit(f"file not found: {sys.argv[1]}")
    for idx, name in m.names.items():
        print(f"{idx}: {name}")


if __name__ == "__main__":
    main()
