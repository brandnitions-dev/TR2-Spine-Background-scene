"""Dump the live western_scene2.psd, rasterize-export, crop, rebuild Spine."""
from __future__ import annotations

import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREV = HERE / "psd2_layers.prev.json"
CUR = HERE / "psd2_layers.json"
JSX = HERE / "_export_psd2_hq.jsx"


def run_jsx(app) -> None:
    jsx = str(JSX).replace("\\", "/")
    last = None
    for method in ("DoJavaScriptFile", "DoJavaScript"):
        try:
            if method == "DoJavaScriptFile":
                app.DoJavaScriptFile(jsx)
            else:
                app.DoJavaScript(JSX.read_text(encoding="utf-8"))
            print(f"JSX ok via {method}")
            return
        except Exception as exc:
            last = exc
            print(f"JSX {method} failed: {exc}")
    raise SystemExit(f"Could not run {JSX}: {last}")


def main() -> int:
    if CUR.is_file():
        shutil.copy2(CUR, PREV)
        print(f"copied prev {PREV}")

    import export_psd2

    export_psd2.main()
    print("layer dump done")

    app = export_psd2.connect_app()
    docs, match = export_psd2.find_scene2(app)
    if match is None:
        raise SystemExit("western_scene2.psd not open after dump")
    app.ActiveDocument = match[0]
    print("running HQ rasterize export; keep Photoshop in front")
    t0 = time.time()
    run_jsx(app)
    print(f"JSX finished in {time.time() - t0:.1f}s")

    import crop_psd2_parts

    crop_psd2_parts.main()
    print("crop done")

    import build_scene

    build_scene.main()
    print("spine rebuild done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
