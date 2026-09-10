"""Legacy entrypoint shim.

The real entrypoints are ``python -m rlmalloc.train`` and
``python -m rlmalloc.evaluate``.  This file preserves ``python main.py``:
by default it runs training; pass ``--mode eval`` for evaluation.
"""

import sys


def main() -> None:
    mode = "train"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        mode = sys.argv[idx + 1]
        del sys.argv[idx:idx + 2]

    if mode == "eval":
        from rlmalloc.evaluate import main as run
    elif mode == "train":
        from rlmalloc.train import main as run
    else:
        raise SystemExit(f"unknown mode {mode!r} (use 'train' or 'eval')")
    run()


if __name__ == "__main__":
    main()
