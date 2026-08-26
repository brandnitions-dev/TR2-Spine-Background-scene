"""Export western_scene2.psd from the saved Desktop file when Photoshop COM is locked."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from psd_tools import PSDImage

HERE = Path(__file__).resolve().parent
DESKTOP = Path(r"C:\Users\Emex33\Desktop\western_scene2.psd")
PSD_COPY = HERE / "western_scene2.psd"
OUT_JSON = HERE / "psd2_layers.json"
PREV = HERE / "psd2_layers.prev.json"
PARTS = HERE / "psd2-parts"
CURVES_JSON = HERE / "curves_1.json"
SKIP_PARENTS = {"board"}
SKIP_NAMES: set[str] = set()
RED_FILTER_NAMES = {"red filter", "red_filter"}
RED_FILTER_JSON = HERE / "red_filter.json"


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


def cubic_spline_lut(points: list[tuple[int, int]]) -> list[int]:
    xs = np.array([float(p[0]) for p in points], dtype=np.float64)
    ys = np.array([float(p[1]) for p in points], dtype=np.float64)
    n = len(xs)
    if n < 2:
        return list(range(256))
    if n == 2:
        return [int(round(min(255, max(0, v)))) for v in np.interp(np.arange(256), xs, ys)]
    h = np.diff(xs)
    alpha = np.zeros(n)
    for i in range(1, n - 1):
        alpha[i] = (3.0 / h[i]) * (ys[i + 1] - ys[i]) - (3.0 / h[i - 1]) * (ys[i] - ys[i - 1])
    el = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    for i in range(1, n - 1):
        el[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / el[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / el[i]
    c = np.zeros(n)
    b = np.zeros(n)
    d = np.zeros(n)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])
    lut: list[int] = []
    i = 0
    for x in range(256):
        while i < n - 2 and x > xs[i + 1]:
            i += 1
        dx = x - xs[i]
        y = ys[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx
        lut.append(int(round(min(255.0, max(0.0, y)))))
    return lut


def photoshop_curve_points(stored: list[tuple[int, int]]) -> list[tuple[int, int]]:
    # psd_tools stores the inverse of the Photoshop Input/Output graph.
    # The live Curves 1 graph sits below the diagonal (darken).
    pts = [(int(p[1]), int(p[0])) for p in stored]
    pts.sort(key=lambda p: (p[0], p[1]))
    cleaned: list[tuple[int, int]] = []
    for x, y in pts:
        if cleaned and cleaned[-1][0] == x:
            cleaned[-1] = (x, y)
        else:
            cleaned.append((x, y))
    if not cleaned or cleaned[0][0] > 0:
        cleaned.insert(0, (0, 0))
    if cleaned[-1][0] < 255:
        cleaned.append((255, 255))
    return cleaned


def write_curves_lut(layer, psd: PSDImage | None = None) -> dict | None:
    data = getattr(layer, "data", None)
    stored = None
    if data is not None and getattr(data, "data", None):
        stored = [tuple(int(v) for v in pair) for pair in data.data[0]]
    extra = getattr(data, "extra", None) if data is not None else None
    if extra:
        item = extra[0]
        extra_pts = getattr(item, "points", None)
        if extra_pts:
            stored = [tuple(int(v) for v in pair) for pair in extra_pts]
    if not stored:
        return None
    points = photoshop_curve_points(stored)
    lut = cubic_spline_lut(points)
    method = "photoshop_graph"
    payload = {
        "name": str(layer.name),
        "on": True,
        "points": [list(p) for p in points],
        "stored": [list(p) for p in stored],
        "lut": lut,
        "method": method,
    }
    CURVES_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {CURVES_JSON} points={payload['points']} method={method}")
    return payload


def export_curves_standalone() -> int:
    src = DESKTOP if DESKTOP.is_file() else PSD_COPY
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    psd = PSDImage.open(src)
    found = None
    for layer in psd.descendants():
        kind = str(getattr(layer, "kind", "") or "")
        if "curve" in str(layer.name).lower() or kind.lower() == "curves":
            found = write_curves_lut(layer, psd)
            break
    if not found:
        raise SystemExit("no Curves layer in PSD")
    dest = HERE / "spine-scene" / "curves_1.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(CURVES_JSON.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"copied {dest}")
    return 0


def is_red_filter_name(name: str) -> bool:
    return name.strip().lower().replace(" ", "_") == "red_filter"


def unique_safe(base: str, used: dict[str, int]) -> str:
    n = used.get(base, 0) + 1
    used[base] = n
    return base if n == 1 else f"{base}_{n:02d}"


def paste_safe(canvas: Image.Image, tile: Image.Image, x0: int, y0: int) -> None:
    tw, th = tile.size
    src_l = max(0, -x0)
    src_t = max(0, -y0)
    dst_l = max(0, x0)
    dst_t = max(0, y0)
    width = min(tw - src_l, canvas.width - dst_l)
    height = min(th - src_t, canvas.height - dst_t)
    if width <= 0 or height <= 0:
        return
    piece = tile.crop((src_l, src_t, src_l + width, src_t + height))
    canvas.paste(piece, (dst_l, dst_t), piece)


def write_red_filter_meta(layer) -> dict:
    opacity_pct = round(float(getattr(layer, "opacity", 255)) * 100.0 / 255.0, 3)
    payload = {
        "name": str(layer.name),
        "blend": "linear_burn",
        "psd_opacity": opacity_pct,
        "base_opacity": 0.45,
        "bonus_opacity": 0.80,
        "mode": "base",
        "export": "full_strength_topil",
        "note": "PSD layer opacity is Base 45%. Viewer Bonus uses 80%. Do not bake either value into the PNG.",
    }
    RED_FILTER_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {RED_FILTER_JSON} psd_opacity={opacity_pct}")
    return payload


def layer_tile(layer) -> Image.Image | None:
    composite = None
    try:
        composite = layer.composite(force=True)
    except Exception as exc:
        print(f"composite failed {layer.name}: {exc}")
    raw = None
    try:
        raw = layer.topil()
    except Exception:
        raw = None
    tile = composite if composite is not None else raw
    if tile is None:
        return None
    tile = tile.convert("RGBA")
    if raw is None:
        return tile
    raw = raw.convert("RGBA")
    composite_alpha = np.asarray(tile)[:, :, 3]
    raw_alpha = np.asarray(raw)[:, :, 3]
    if int((composite_alpha > 8).sum()) == 0 and int((raw_alpha > 8).sum()) > 0:
        return raw
    return tile


def layer_tile_for_export(layer, name: str) -> Image.Image | None:
    if name.strip().lower() in RED_FILTER_NAMES:
        raw = None
        try:
            raw = layer.topil()
        except Exception:
            raw = None
        if raw is not None:
            return raw.convert("RGBA")
        tile = layer_tile(layer)
        if tile is None:
            return None
        op = float(getattr(layer, "opacity", 255)) / 255.0
        if op > 0.01 and op < 0.999:
            arr = np.asarray(tile).copy()
            arr[:, :, 3] = np.clip(arr[:, :, 3].astype(np.float32) / op, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, "RGBA")
        return tile
    return layer_tile(layer)


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
        kind_name = str(getattr(layer, "kind", "") or "")
        is_curves = (not is_group) and ("curve" in name.lower() or kind_name.lower() == "curves")
        rec = {
            "index": index,
            "stack_index": index,
            "name": name,
            "safe_name": sanitize(name, index),
            "parent": parent,
            "depth": depth_of(layer),
            "kind": "group" if is_group else ("curves" if is_curves else "art"),
            "ps_kind": "group" if is_group else ("curves" if is_curves else "raster"),
            "visible": bool(layer.visible),
            "opacity": round(float(getattr(layer, "opacity", 255)) * 100.0 / 255.0, 3),
            "blend": str(getattr(layer, "blend_mode", "normal")).split(".")[-1].lower(),
            "bounds": [float(v) for v in bbox] if bbox else None,
        }
        if bbox and len(bbox) == 4:
            rec["width"] = round(float(bbox[2] - bbox[0]), 3)
            rec["height"] = round(float(bbox[3] - bbox[1]), 3)
        layers.append(rec)
        if is_curves:
            write_curves_lut(layer, psd)
            continue
        if name.strip().lower() in RED_FILTER_NAMES:
            write_red_filter_meta(layer)
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
        if (not bool(layer.visible)) and (name.strip().lower() not in RED_FILTER_NAMES):
            log.append(f"skip hidden {name!r} parent={parent!r}")
            continue
        jsx = unique_safe(sanitize(name, index), used)
        tile = layer_tile_for_export(layer, name)
        if tile is None:
            log.append(f"skip {jsx} no composite")
            continue
        x0 = int(round(bbox[0]))
        y0 = int(round(bbox[1]))
        dest = PARTS / f"full_{jsx}.png"
        keep_native = "cloud" in name.lower()
        if keep_native:
            tile.save(dest)
        else:
            canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            paste_safe(canvas, tile, x0, y0)
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
