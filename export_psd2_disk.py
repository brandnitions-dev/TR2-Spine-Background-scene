"""Export western_scene2.psd from the saved Desktop file when Photoshop COM is locked."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image
from psd_tools import PSDImage

HERE = Path(__file__).resolve().parent
DESKTOP = Path(r"C:\Users\Emex33\Desktop\western_scene2.psd")
PSD_COPY = HERE / "western_scene2.psd"
OUT_JSON = HERE / "psd2_layers.json"
PREV = HERE / "psd2_layers.prev.json"
PARTS = HERE / "psd2-parts"
SKIP_PARENTS = {"board"}
SKIP_NAMES = {"red filter", "red_filter"}


def sanitize(name: str, index: int) -> str:
    out = []
    for ch in name.strip().lower().replace(" ", "_"):
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    safe = "".join(out).strip("_") or f"layer_{index:02d}"
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


def parent_name(layer) -> str:
    parent = getattr(layer, "parent", None)
    if parent is None or not hasattr(parent, "name"):
        return ""
    name = str(parent.name)
    return "" if name.lower() == "root" else name


def depth_of(layer) -> int:
    n = 0
    p = getattr(layer, "parent", None)
    while p is not None and hasattr(p, "name") and str(p.name).lower() != "root":
        n += 1
        p = getattr(p, "parent", None)
    return n


def unique_safe(base: str, used: dict[str, int]) -> str:
    n = used.get(base, 0) + 1
    used[base] = n
    return base if n == 1 else f"{base}_{n:02d}"


def main() -> int:
    if not DESKTOP.is_file():
        raise SystemExit(f"missing {DESKTOP}")
    if OUT_JSON.is_file():
        shutil.copy2(OUT_JSON, PREV)
    shutil.copy2(DESKTOP, PSD_COPY)
    psd = PSDImage.open(DESKTOP)
    canvas_w, canvas_h = int(psd.width), int(psd.height)
    PARTS.mkdir(parents=True, exist_ok=True)
    for old in PARTS.glob("full_*.png"):
        old.unlink()
    for old in PARTS.glob("stack_*.png"):
        old.unlink()

    layers: list[dict] = []
    log = [f"doc={DESKTOP.name}", f"src_canvas={canvas_w}x{canvas_h}", f"export_canvas={canvas_w}x{canvas_h}", "scale=1", "effects=psd_tools_composite", "smart_objects=composite"]
    used: dict[str, int] = {}
    index = 0
    for layer in psd.descendants():
        index += 1
        is_group = layer.is_group()
        bbox = list(layer.bbox) if layer.bbox else None
        name = str(layer.name)
        parent = parent_name(layer)
        rec = {
            "index": index,
            "stack_index": index,
            "name": name,
            "safe_name": sanitize(name, index),
            "parent": parent,
            "depth": depth_of(layer),
            "kind": "group" if is_group else "art",
            "ps_kind": "group" if is_group else "raster",
            "visible": bool(layer.visible),
            "opacity": round(float(getattr(layer, "opacity", 255)) * 100.0 / 255.0, 3),
            "blend": str(getattr(layer, "blend_mode", "normal")).split(".")[-1].lower(),
            "bounds": [float(v) for v in bbox] if bbox else None,
        }
        if bbox and len(bbox) == 4:
            rec["width"] = round(float(bbox[2] - bbox[0]), 3)
            rec["height"] = round(float(bbox[3] - bbox[1]), 3)
        layers.append(rec)
        if is_group or not bbox:
            continue
        skip = name.strip().lower() in SKIP_NAMES or parent.lower() in SKIP_PARENTS
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        if w < 1 or h < 1:
            log.append(f"skip {sanitize(name, index)} empty")
            continue
        if skip:
            log.append(f"skip {name!r} parent={parent!r}")
            continue
        jsx = unique_safe(sanitize(name, index), used)
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        try:
            tile = layer.composite(force=True)
        except Exception as exc:
            print(f"composite failed {name}: {exc}")
            continue
        if tile is None:
            log.append(f"skip {jsx} no composite")
            continue
        tile = tile.convert("RGBA")
        x0 = int(round(bbox[0]))
        y0 = int(round(bbox[1]))
        canvas.paste(tile, (x0, y0), tile)
        dest = PARTS / f"full_{jsx}.png"
        canvas.save(dest)
        log.append(
            f"full {jsx} {bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]} name={name}"
        )
        log.append(
            f"meta {jsx} kind=raster vis={str(bool(layer.visible)).lower()} "
            f"op={layer.opacity} blend={getattr(layer, 'blend_mode', 'normal')} "
            f"canvas_bounds={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]} name={name}"
        )
        print(f"full {jsx} {dest.name} {tile.size} @ {x0},{y0}")

    payload = {
        "source_open": DESKTOP.name,
        "source_full": str(DESKTOP),
        "saved_copy": str(PSD_COPY),
        "canvas": {"w": canvas_w, "h": canvas_h},
        "mode": "disk_psd_tools",
        "layer_count": len(layers),
        "layers": layers,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (PARTS / "export_log.txt").write_text("\n".join(log), encoding="utf-8")
    print(f"wrote {OUT_JSON} layers={len(layers)} canvas={canvas_w}x{canvas_h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
