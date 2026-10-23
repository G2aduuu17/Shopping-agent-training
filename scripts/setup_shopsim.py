"""Prepare only the pinned CPU environment and index; never install upstream training extras."""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from shopmini.artifacts import sha256, write_json
from shopmini.upstream import MINI_ROOT, PRODUCT_SHA256, UPSTREAM_COMMIT, require_upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Download Python 3.10 and environment dependencies")
    args = parser.parse_args()
    root = require_upstream()
    shop = root / "environments/ShopSimulator/shop_env"
    runtime = MINI_ROOT / "runtime/shopsim"
    runtime.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(UV_CACHE_DIR=str(MINI_ROOT / ".cache/uv"),
               UV_PYTHON_INSTALL_DIR=str(MINI_ROOT / ".python"), PYTHONUTF8="1")
    uv = Path(sys.executable).parent / ("uv.exe" if os.name == "nt" else "uv")
    python = MINI_ROOT / ".venv-shopsim" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if args.install:
        if not uv.exists():
            raise RuntimeError(f"Install uv first: {sys.executable} -m pip install uv")
        if not python.exists():
            subprocess.run([str(uv), "venv", "--python", "3.10.21", str(python.parents[1])], env=env, check=True)
        subprocess.run([str(uv), "pip", "install", "--python", str(python), "-r",
                        str(MINI_ROOT / "requirements-shopsim-windows.lock.txt")], env=env, check=True)
    if not python.exists():
        raise RuntimeError("Run setup_shopsim.py --install first")
    # Match upstream's expected path without changing tracked upstream source.
    product = shop / "data/items_eval_train.json"
    if not product.exists():
        staging = product.with_suffix(".preparing")
        with (gzip.open(shop / "data/fine_items_eval_train_all.json.gz", "rb") as source,
              staging.open("wb") as dest):
            shutil.copyfileobj(source, dest)
        if sha256(staging) != PRODUCT_SHA256:
            raise RuntimeError("Product checksum mismatch; staging retained for inspection")
        os.replace(staging, product)
    if sha256(product) != PRODUCT_SHA256:
        raise RuntimeError("Existing product checksum mismatch")
    index = runtime / "products.sqlite3"
    metadata = runtime / "products.manifest.json"
    rebuild = not index.exists() or not metadata.exists()
    if not rebuild:
        previous = json.loads((runtime / "setup.json").read_text()) if (runtime / "setup.json").exists() else {}
        rebuild = (previous.get("index_sha256") != sha256(index)
                   or previous.get("upstream_commit") != UPSTREAM_COMMIT)
    if rebuild:
        subprocess.run([str(python), str(shop / "scripts/build_index.py"), "--products", str(product),
                        "--output", str(index), "--manifest", str(metadata)],
                       cwd=shop, env=env, check=True)
    report = {"upstream_commit": UPSTREAM_COMMIT, "product_sha256": PRODUCT_SHA256,
              "index_sha256": sha256(index), "index_manifest": json.loads(metadata.read_text()),
              "environment_config_sha256": sha256(shop / "configs/environment.json"),
              "requirements_sha256": sha256(shop / "requirements.txt"),
              "service_lock_sha256": sha256(MINI_ROOT / "requirements-shopsim-windows.lock.txt"),
              "python": str(python)}
    write_json(runtime / "setup.json", report)
    print(f"ShopSimulator ready. Setup manifest: {runtime / 'setup.json'}")


if __name__ == "__main__":
    main()
