"""Crop full-canvas PSD exports to effect-aware bounds and prefer native SO pixels."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PARTS = HERE / "psd2-parts"
JSON_PATH = HERE / "psd2_layers.json"
LOG_PATH = PARTS / "export_log.txt"
HQ_DUMP = PARTS / "hq_dump.txt"
ALPHA_CUT = 12


def sanitize_jsx(name: str) -> str:
    out = []
    for ch in name.strip().lower():
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    safe = "".join(out).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


def slot_name(name: str) -> str:
    out = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
        else:
            out.append("_")
    safe = "".join(out).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or "layer"


def alpha_metrics(arr: np.ndarray) -> dict:
    alpha = arr[:, :, 3]
    mask = alpha > ALPHA_CUT
    ys, xs = np.where(mask)
    h, w = alpha.shape
    if xs.size == 0:
        return {
            "bbox": [0, 0, w, h],
            "hang": [w / 2.0, 0.0],
            "bottom": [w / 2.0, float(max(h - 1, 0))],
            "center": [w / 2.0, h / 2.0],
            "opaque": 0,
            "mean_rgb": [0, 0, 0],
            "mean_luma": 0.0,
            "mean_warmth": 0.0,
        }
    r = arr[:, :, 0][mask].astype(np.float32)
    g = arr[:, :, 1][mask].astype(np.float32)
    b = arr[:, :, 2][mask].astype(np.float32)
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    warmth = r - b
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    top = ys == y0
    bot = ys == y1
    return {
        "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
        "hang": [float(xs[top].mean()), float(y0)],
        "bottom": [float(xs[bot].mean()), float(y1)],
        "center": [float(xs.mean()), float(ys.mean())],
        "opaque": int(mask.sum()),
        "mean_rgb": [float(r.mean()), float(g.mean()), float(b.mean())],
        "mean_luma": float(luma.mean()),
        "mean_warmth": float(warmth.mean()),
    }


def parse_export_log(text: str) -> dict:
    info: dict = {
        "scale": 1.0,
        "src_canvas": None,
        "export_canvas": None,
        "full": {},
        "so": {},
        "meta": {},
    }
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("scale="):
            info["scale"] = float(line.split("=", 1)[1])
        elif line.startswith("src_canvas="):
            info["src_canvas"] = line.split("=", 1)[1]
        elif line.startswith("export_canvas="):
            info["export_canvas"] = line.split("=", 1)[1]
        elif line.startswith("full "):
            m = re.match(
                r"full (\S+) (-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)(?: name=(.*))?$",
                line,
            )
            if m:
                info["full"][m.group(1)] = {
                    "bounds": [float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))],
                    "name": (m.group(6) or m.group(1)).strip(),
                }
        elif line.startswith("so "):
            m = re.match(r"so (\S+) (\d+)x(\d+) ok=(true|false)", line)
            if m:
                info["so"][m.group(1)] = {
                    "w": int(m.group(2)),
                    "h": int(m.group(3)),
                    "ok": m.group(4) == "true",
                }
        elif line.startswith("meta "):
            m = re.match(
                r"meta (\S+) kind=(\S+) vis=(\S+) op=(\S+) blend=(\S+) canvas_bounds=(-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+) name=(.*)$",
                line,
            )
            if m:
                info["meta"][m.group(1)] = {
                    "kind": m.group(2),
                    "visible": m.group(3) == "true",
                    "opacity": m.group(4),
                    "blend": m.group(5),
                    "canvas_bounds": [
                        float(m.group(6)),
                        float(m.group(7)),
                        float(m.group(8)),
                        float(m.group(9)),
                    ],
                    "name": m.group(10).strip(),
                }
    return info


def parse_hq_dump(text: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for raw in text.splitlines():
        if not raw.startswith("LAYER|"):
            continue
        parts = raw.split("|")
        name = parts[1]
        rec: dict = {"psd_name": name}
        for part in parts[2:]:
            if "=" not in part:
                continue
            key, val = part.split("=", 1)
            rec[key] = val
        rows[sanitize_jsx(name)] = rec
    return rows


def canvas_box(bounds: list[float]) -> list[int]:
    left, top, right, bottom = bounds
    return [
        int(np.floor(left)),
        int(np.floor(top)),
        int(np.ceil(right) - np.floor(left)),
        int(np.ceil(bottom) - np.floor(top)),
    ]


def crop_box(im: Image.Image, bounds: list[float]) -> tuple[Image.Image, list[int]]:
    left, top, right, bottom = bounds
    cl = max(0, int(np.floor(left)))
    ct = max(0, int(np.floor(top)))
    cr = min(im.width, int(np.ceil(right)))
    cb = min(im.height, int(np.ceil(bottom)))
    placed = canvas_box(bounds)
    if cr <= cl or cb <= ct:
        return im, placed
    return im.crop((cl, ct, cr, cb)), placed


def flatten_darken_appearance(iso: Image.Image, stack_path: Path) -> Image.Image:
    """Use composite RGB (Darken already applied) and the isolated layer alpha."""
    stack = Image.open(stack_path).convert("RGBA")
    if stack.size != iso.size:
        stack = stack.resize(iso.size, Image.Resampling.NEAREST)
    iso_a = np.asarray(iso)[:, :, 3]
    arr = np.asarray(stack).copy()
    arr[:, :, 3] = np.minimum(arr[:, :, 3], iso_a)
    return Image.fromarray(arr, "RGBA")


CROP_ALPHA_CUT = 1


def alpha_crop(im: Image.Image, fallback_bounds: list[float]) -> tuple[Image.Image, list[int]]:
    arr = np.asarray(im)
    ys, xs = np.where(arr[:, :, 3] > CROP_ALPHA_CUT)
    if xs.size == 0:
        return crop_box(im, fallback_bounds)
    left = max(0, int(xs.min()))
    top = max(0, int(ys.min()))
    right = min(im.width, int(xs.max()) + 1)
    bottom = min(im.height, int(ys.max()) + 1)
    return im.crop((left, top, right, bottom)), [left, top, right - left, bottom - top]


def choose_source(
    jsx: str,
    layer_name: str,
    full_im: Image.Image,
    effects_bounds: list[float],
    canvas_w: int,
    canvas_h: int,
    log: dict,
) -> tuple[Image.Image, list[int], float, str, dict]:
    extra = {
        "effects_bounds": effects_bounds,
        "glow_expanded": False,
    }
    meta = log["meta"].get(jsx) or {}
    canvas_bounds = meta.get("canvas_bounds")
    appearance = full_im
    source = "raster_isolated"
    stack_path = PARTS / f"stack_{jsx}.png"
    if stack_path.is_file():
        appearance = flatten_darken_appearance(full_im, stack_path)
        source = "flattened_blend_appearance"

    cropped, placed = alpha_crop(appearance, effects_bounds)
    extra["src_px"] = list(cropped.size)
    if canvas_bounds:
        extra["psd_bounds_no_effects"] = canvas_bounds
        dw = placed[2] - (canvas_bounds[2] - canvas_bounds[0])
        dh = placed[3] - (canvas_bounds[3] - canvas_bounds[1])
        extra["glow_expanded"] = dw > 0.5 or dh > 0.5
        extra["glow_delta"] = [round(dw, 2), round(dh, 2)]
    return cropped, placed, 1.0, source, extra


def main() -> int:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    canvas_w = int(payload["canvas"]["w"])
    canvas_h = int(payload["canvas"]["h"])
    log = parse_export_log(LOG_PATH.read_text(encoding="utf-8")) if LOG_PATH.is_file() else {
        "scale": 1.0,
        "full": {},
        "so": {},
        "meta": {},
    }
    dump = parse_hq_dump(HQ_DUMP.read_text(encoding="utf-8")) if HQ_DUMP.is_file() else {}
    export_scale = float(log.get("scale") or 1.0)
    used: dict[str, int] = {}
    jsx_used: dict[str, int] = {}
    exported: list[dict] = []

    SKIP_PARENTS = {"board"}
    SKIP_NAMES = {"curves 1"}
    RED_FILTER_NAMES = {"red filter", "red_filter"}
    BG_PARENTS = {"background layer"}

    def is_red_filter_layer(layer: dict, slot: str = "") -> bool:
        raw = (layer.get("name") or slot or "").strip().lower().replace(" ", "_")
        return raw == "red_filter"

    def is_bg_source(layer: dict, slot: str) -> bool:
        if is_red_filter_layer(layer, slot):
            return False
        parent = (layer.get("parent") or "").strip().lower()
        raw = (layer.get("name") or "").strip().lower()
        return parent in BG_PARENTS or "background" in raw or "cloud" in raw or "cloud" in slot.lower()

    for layer in payload["layers"]:
        if layer["kind"] != "art":
            continue
        bounds = layer.get("bounds")
        if not bounds:
            continue
        raw_name = (layer.get("name") or "").strip()
        parent = (layer.get("parent") or "").strip().lower()
        if raw_name.strip().lower() in SKIP_NAMES or parent in SKIP_PARENTS:
            print(f"skip hidden-or-stale {raw_name!r} parent={layer.get('parent')!r}")
            continue
        if (not bool(layer.get("visible", True))) and raw_name.strip().lower() not in RED_FILTER_NAMES:
            print(f"skip hidden {raw_name!r} parent={layer.get('parent')!r}")
            continue
        base = sanitize_jsx(layer["name"])
        k = jsx_used.get(base, 0) + 1
        jsx_used[base] = k
        jsx = base if k == 1 else f"{base}_{k:02d}"
        src = PARTS / f"full_{jsx}.png"
        if not src.is_file():
            print(f"skip missing export {src}")
            continue
        im = Image.open(src).convert("RGBA")
        full_rec = log["full"].get(jsx)
        effects = list(full_rec["bounds"]) if full_rec else [float(v) * export_scale for v in bounds]
        if export_scale != 1.0:
            effects = [v / export_scale for v in effects]
            if im.size != (int(canvas_w * export_scale), int(canvas_h * export_scale)):
                print(f"WARNING {src.name} size {im.size} != {canvas_w * export_scale}x{canvas_h * export_scale}")
        elif im.size != (canvas_w, canvas_h):
            print(f"WARNING {src.name} size {im.size} != canvas {canvas_w}x{canvas_h}")

        cropped, placed, scale, source, extra = choose_source(
            jsx, layer["name"], im, effects, canvas_w, canvas_h, log
        )
        if im.size != (canvas_w, canvas_h):
            cropped, img_placed = alpha_crop(im, effects)
            placed = [
                int(np.floor(effects[0])) + img_placed[0],
                int(np.floor(effects[1])) + img_placed[1],
                img_placed[2],
                img_placed[3],
            ]
            extra["src_px"] = list(cropped.size)
            extra["native_overflow"] = True
        width, height = placed[2], placed[3]
        if width < 1 or height < 1:
            print(f"skip {layer['name']} clamped empty")
            continue
        name = slot_name(layer["name"])
        if is_red_filter_layer(layer, name):
            name = "red_filter"
        if is_bg_source(layer, name):
            if "cloud" in name.lower():
                name = "background_clouds"
            elif name.lower() in {"background", "background_redroom"}:
                name = "background"
            elif not name.lower().startswith("background"):
                name = f"background_{name}"
        n = used.get(name, 0) + 1
        used[name] = n
        if n > 1:
            name = f"{name}_{n:02d}"
        dest = PARTS / f"{name}.png"
        keep_full = (
            (is_bg_source(layer, name) and im.size == (canvas_w, canvas_h))
            or raw_name.strip().lower() in RED_FILTER_NAMES
            or name.lower() in RED_FILTER_NAMES
        )
        if keep_full and im.size == (canvas_w, canvas_h):
            im.save(dest)
            native = [canvas_w, canvas_h]
            placed = [0, 0, canvas_w, canvas_h]
            scale = 1.0
        else:
            cropped.save(dest)
            native = list(cropped.size)

        arr = np.asarray(Image.open(dest).convert("RGBA"))
        metrics = alpha_metrics(arr)
        dump_rec = dump.get(jsx) or {}
        meta = log["meta"].get(jsx) or {}
        rec = {
            "psd_name": layer["name"],
            "name": name,
            "filename": dest.name,
            "parent_group": layer.get("parent") or "",
            "x": placed[0],
            "y": placed[1],
            "w": placed[2],
            "h": placed[3],
            "scale": float(scale),
            "native_w": native[0],
            "native_h": native[1],
            "src_px": extra.get("src_px") or native,
            "canvas_bounds": extra.get("psd_bounds_no_effects") or bounds,
            "effects_bounds": extra.get("effects_bounds") or effects,
            "glow_expanded": bool(extra.get("glow_expanded")),
            "glow_delta": extra.get("glow_delta") or [0, 0],
            "layer_kind": meta.get("kind") or dump_rec.get("kind") or "raster",
            "source": source,
            "psd_bounds": bounds,
            "opacity": layer.get("opacity"),
            "blend": layer.get("blend"),
            "stack_index": layer.get("index"),
            "metrics": metrics,
        }
        if extra.get("so_px"):
            rec["so_px"] = extra["so_px"]
        exported.append(rec)
        print(
            f"{name:22s} {native[0]:4d}x{native[1]:<4d} place={placed[2]}x{placed[3]} "
            f"scale={scale:.3f} src={source} glow={rec['glow_expanded']} "
            f"kind={rec['layer_kind']} opaque={metrics['opaque']:7d}"
        )

    posts = [p for p in exported if p["name"].startswith("post_R") or p["name"] == "post_L"]
    if posts and not any(p["name"] == "post_L" for p in exported):
        left = min(posts, key=lambda p: float(p["x"]))
        old = PARTS / left["filename"]
        new_name = "post_L"
        dest = PARTS / f"{new_name}.png"
        if old != dest and old.is_file():
            old.replace(dest)
        left["name"] = new_name
        left["filename"] = dest.name
        print(f"renamed leftmost post {left['psd_name']!r} -> post_L")

    payload.pop("kept_exports", None)
    payload["exports"] = exported
    payload["export_method"] = {
        "scale": export_scale,
        "effects": "rasterize_layerstyle+entirelayer",
        "smart_objects": "native PNG when larger than on-canvas crop",
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(exported)} cropped parts -> {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
