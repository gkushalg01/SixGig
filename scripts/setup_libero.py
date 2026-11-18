"""Write LIBERO's config file non-interactively.

hf-libero prompts on stdin the first time `libero.libero` is imported, which breaks
scripted and headless runs. This writes the same default config the prompt would,
without importing the package.

    python scripts/setup_libero.py           # write ~/.libero/config.yaml if missing
    python scripts/setup_libero.py --force   # overwrite it
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def default_paths() -> dict[str, str]:
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("libero is not installed; install lerobot[libero] first")
    root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    # Same keys and layout as libero.libero.get_default_path_dict().
    return {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str((root / ".." / "datasets").resolve()),
        "assets": str(root / "assets"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite an existing config")
    args = parser.parse_args(argv)

    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    config_file = config_dir / "config.yaml"
    if config_file.exists() and not args.force:
        print(f"{config_file} already exists; use --force to overwrite")
        return 0
    config_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {value}" for key, value in default_paths().items()]
    config_file.write_text("\n".join(lines) + "\n")
    print(f"wrote {config_file}:\n  " + "\n  ".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
