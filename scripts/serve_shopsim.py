"""Launch pinned upstream Flask service on localhost, using its isolated CPU interpreter."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5700)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not 1 <= args.slots <= 8:
        parser.error("port must be 1024..65535; slots must be 1..8")
    mini = Path(__file__).resolve().parents[1]
    runtime = mini / "runtime/shopsim"
    if not args.child:
        from shopmini.upstream import require_upstream

        require_upstream()
        python = mini / ".venv-shopsim" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.exists() or not (runtime / "setup.json").exists():
            raise RuntimeError("Run setup_shopsim.py --install first")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        subprocess.run([str(python), str(Path(__file__).resolve()), "--child", "--port",
                        str(args.port), "--slots", str(args.slots)], env=env, check=True)
        return
    from shopmini.upstream import require_upstream

    shop = require_upstream() / "environments/ShopSimulator/shop_env"
    os.environ.update(SHOP_ENVIRONMENT_VERSION="shopsimulator-environment-v2.1",
                      SHOP_ENV_CONFIG=str(shop / "configs/environment.json"),
                      SHOP_SEARCH_INDEX=str(runtime / "products.sqlite3"),
                      SHOP_MAX_STEPS="35", SHOPSIM_ENV_SLOTS=str(args.slots), SHOPSIM_PORT=str(args.port))
    sys.path[:0] = [str(shop / "shop_env"), str(shop)]
    os.chdir(runtime)  # Keep server/session logs out of tracked upstream source.
    import pack_api

    pack_api.initialize_environments()
    pack_api.app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
