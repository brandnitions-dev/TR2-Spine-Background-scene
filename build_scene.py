"""
Build a Spine 3.8.75 western scene from western_scene2.psd layer bounds.

Photoshop bounds in psd2_layers.json are ground truth. Do not template-match.

  python backgroundSPINE/export_psd2.py
  python backgroundSPINE/crop_psd2_parts.py
  python backgroundSPINE/build_scene.py
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RIG = ROOT / "rig-framework"
PARTS_DIR = HERE / "psd2-parts"
PSD2_PATH = HERE / "psd2_layers.json"
PLACEMENT_PATH = HERE / "placement.json"
OUT_DIR = HERE / "spine-scene"
IMAGES_DIR = OUT_DIR / "images"
VIEWER_SRC = HERE / "scene-viewer.html"

if str(RIG) not in sys.path:
    sys.path.insert(0, str(RIG))
from rig_framework.keys import (  # noqa: E402
    color_hex,
    keyed_color,
    keyed_rotate_pendulum,
)
from rig_framework.verify import resolve_world_bones  # noqa: E402

SPINE_VERSION = "3.8.75"
IDLE_DURATION = 2.4
FLICKER_SRC_DUR = 4.8
ALPHA_CUT = 12
CANVAS_W = 1536
CANVAS_H = 1024
GRAVITY_PX = 980.0
PENDULUM_AMP = (6.5, 8.0)
PENDULUM_TIME_SCALE = 2.12
PENDULUM_PERIOD_SCALE = (1.0, 1.01)
HOOK_NEAR_X = 80.0
HOOK_NEAR_Y = 56.0
PREV_JSON = HERE / "psd2_layers.prev.json"

LAMP_TOKENS = ("lantern", "lamp")
LIGHT_TOKENS = ("light", "glow")
HANG_TOKENS = ("hang", "hanging")
SIGN_RENAME = {
    "beware1": "sign_beware_1",
    "beware2": "sign_beware_2",
    "deadman": "sign_dead_men",
    "tellnotales": "sign_tell_no_tales",
    "postsign": "sign_highlight",
}
SIGN_WORDS = [
    {"id": "beware1", "slot": "sign_beware_1", "label": "Beware"},
    {"id": "beware2", "slot": "sign_beware_2", "label": "Beware"},
    {"id": "dead_men", "slot": "sign_dead_men", "label": "Dead Men"},
    {"id": "tell_no_tales", "slot": "sign_tell_no_tales", "label": "Tell No Tales"},
]
SIGN_HIGHLIGHT_SLOT = "sign_highlight"
SIGN_STATE_PATH = HERE / "sign_state.json"
LAMP_STATE_PATH = HERE / "lamp_state.json"
SITTING_GLOW_SIZE = 420


def canvas_to_spine(cx: float, cy: float) -> tuple[float, float]:
    return float(cx), float(CANVAS_H - cy)


def attach_offset(width: float, height: float, jx: float, jy: float) -> tuple[float, float]:
    return (width / 2.0) - jx, jy - (height / 2.0)


def alpha_metrics(arr: np.ndarray) -> dict:
    alpha = arr[:, :, 3]
    mask = alpha > ALPHA_CUT
    ys, xs = np.where(mask)
    if xs.size == 0:
        h, w = alpha.shape
        return {
            "bbox": [0, 0, w, h],
            "hang": [w / 2.0, 0.0],
            "bottom": [w / 2.0, float(max(h - 1, 0))],
            "center": [w / 2.0, h / 2.0],
            "opaque": 0,
        }
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
    }


def flame_mask(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    a = arr[:, :, 3]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    warmth = r - b
    h, w = a.shape
    xs = np.arange(w, dtype=np.float32)[None, :]
    central = np.abs(xs - (w * 0.5)) < (w * 0.32)
    core = (a > 24) & (luma > 155) & (warmth > 55) & (r > 155) & central
    mid = (a > 20) & (luma > 120) & (warmth > 45) & (r > 130) & central
    return core, mid


def _radial_glow(
    h: int,
    w: int,
    cx: float,
    cy: float,
    radius: float,
    alpha: float,
    rgb: tuple[int, int, int] = (255, 168, 48),
) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    fall = np.clip(1.0 - dist / max(radius, 1.0), 0.0, 1.0)
    fall = fall ** 1.55
    glow = np.zeros((h, w, 4), dtype=np.uint8)
    glow[:, :, 0] = rgb[0]
    glow[:, :, 1] = rgb[1]
    glow[:, :, 2] = rgb[2]
    glow[:, :, 3] = np.clip(fall * (alpha * 255.0), 0, 255).astype(np.uint8)
    return glow


def make_sitting_glow() -> Image.Image:
    size = SITTING_GLOW_SIZE
    cx = cy = size / 2.0
    acc = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    layers = (
        (210, 0.62, (255, 132, 24)),
        (150, 0.92, (255, 168, 48)),
        (78, 1.00, (255, 214, 110)),
        (30, 1.00, (255, 244, 196)),
    )
    for radius, alpha, rgb in layers:
        tile = Image.fromarray(_radial_glow(size, size, cx, cy, radius, alpha, rgb), "RGBA")
        acc = Image.alpha_composite(acc, tile)
    return acc.filter(ImageFilter.GaussianBlur(radius=2.6))


def extract_lamp_layers(src: Path, body_path: Path, glow_path: Path, *, force_on: bool = False) -> dict:
    im = Image.open(src).convert("RGBA")
    arr = np.asarray(im).copy()
    h, w = arr.shape[:2]
    core, mid = flame_mask(arr)
    core_n = int(core.sum())
    mid_n = int(mid.sum())
    method = "extracted+halo"
    if core_n >= 40:
        ys, xs = np.where(core)
        cx, cy = float(xs.mean()), float(ys.mean())
        halo_r = max(w, h) * 0.20
        halo_a = 0.72
        body = arr.copy()
        dim = core | mid
        body[dim, 0] = (body[dim, 0].astype(np.float32) * 0.42).astype(np.uint8)
        body[dim, 1] = (body[dim, 1].astype(np.float32) * 0.38).astype(np.uint8)
        body[dim, 2] = (body[dim, 2].astype(np.float32) * 0.35).astype(np.uint8)
        glow = _radial_glow(h, w, cx, cy, halo_r, halo_a)
        extracted = np.zeros_like(arr)
        extracted[mid] = arr[mid]
        extracted[core] = arr[core]
        glow_im = Image.fromarray(glow, "RGBA")
        ext_im = Image.fromarray(extracted, "RGBA")
        glow_im = Image.alpha_composite(glow_im, ext_im)
        glow_im = glow_im.filter(ImageFilter.GaussianBlur(radius=1.2))
        glow_size = [w, h]
    else:
        method = "forced_on_bright" if force_on else "synthetic_halo"
        metrics = alpha_metrics(arr)
        cx = metrics["center"][0]
        cy = metrics["hang"][1] + (h * 0.55)
        if mid_n:
            ys, xs = np.where(mid)
            cx, cy = float(xs.mean()), float(ys.mean())
        body = arr.copy()
        if force_on:
            glow_im = make_sitting_glow()
            glow_size = [SITTING_GLOW_SIZE, SITTING_GLOW_SIZE]
        else:
            radius = max(w, h) * 0.18
            glow_im = Image.fromarray(_radial_glow(h, w, cx, cy, radius, 0.55), "RGBA")
            glow_im = glow_im.filter(ImageFilter.GaussianBlur(radius=1.6))
            glow_size = [w, h]
        core_n = 0

    Image.fromarray(body, "RGBA").save(body_path)
    glow_im.save(glow_path)
    garr = np.asarray(glow_im)
    return {
        "method": method,
        "core_px": core_n,
        "mid_px": mid_n,
        "glow_opaque": int((garr[:, :, 3] > 8).sum()),
        "size": [w, h],
        "glow_size": glow_size,
        "flame_img": [round(float(cx), 2), round(float(cy), 2)],
        "bright": bool(force_on),
    }


def own_name(part: dict) -> str:
    return f"{part.get('name') or ''} {part.get('psd_name') or ''}".lower()


def is_background(part: dict) -> bool:
    text = f"{part.get('name') or ''} {part.get('psd_name') or ''}".lower()
    return "background" in text


def is_baked_light(part: dict) -> bool:
    name = (part.get("name") or "").lower()
    psd = (part.get("psd_name") or "").lower()
    return name in LIGHT_TOKENS or psd in LIGHT_TOKENS


def is_lamp(part: dict) -> bool:
    if is_baked_light(part):
        return False
    return any(tok in own_name(part) for tok in LAMP_TOKENS)


def is_hanging_lamp(part: dict) -> bool:
    if not is_lamp(part):
        return False
    text = own_name(part)
    if any(tok in text for tok in HANG_TOKENS):
        return True
    return float(part["y"]) < (CANVAS_H * 0.35)


def sign_psd_key(part: dict) -> str:
    raw = (part.get("psd_name") or part.get("name") or "").strip().lower()
    return "".join(ch for ch in raw if ch.isalnum())


def apply_sign_names(parts: list[dict]) -> list[dict]:
    out: list[dict] = []
    for part in parts:
        rec = dict(part)
        key = sign_psd_key(rec)
        if key in SIGN_RENAME:
            rec["name"] = SIGN_RENAME[key]
        out.append(rec)
    return out


def is_sign_overlay(part: dict) -> bool:
    name = part.get("name") or ""
    if name == "signpost":
        return False
    if name in {w["slot"] for w in SIGN_WORDS} or name == SIGN_HIGHLIGHT_SLOT:
        return True
    group = (part.get("parent_group") or "").lower()
    return group == "group 1" and name != "signpost"


def default_sign_state() -> dict:
    words = {w["id"]: True for w in SIGN_WORDS}
    return apply_sign_state(words)


def apply_sign_state(words_on: dict) -> dict:
    words = {w["id"]: bool(words_on.get(w["id"], False)) for w in SIGN_WORDS}
    highlight = any(words.values())
    slots = {w["slot"]: words[w["id"]] for w in SIGN_WORDS}
    slots[SIGN_HIGHLIGHT_SLOT] = highlight
    return {
        "beware1": words["beware1"],
        "beware2": words["beware2"],
        "dead_men": words["dead_men"],
        "tell_no_tales": words["tell_no_tales"],
        "highlight": highlight,
        "highlight_rule": "OR: highlight on if any word is on",
        "words": SIGN_WORDS,
        "highlight_slot": SIGN_HIGHLIGHT_SLOT,
        "slots": slots,
    }


def default_lamp_state(sitting: list[str]) -> dict:
    labels = {"lantern_dim": "Barrel lamp"}
    lamps = []
    for name in sitting:
        lamps.append(
            {
                "id": name,
                "slot": f"{name}_light",
                "label": labels.get(name, name.replace("_", " ").title()),
                "on": True,
            }
        )
    return {"lamps": lamps}


def overlay_attach_on_bone(part: dict, bone_world: tuple[float, float]) -> tuple[float, float]:
    cx = float(part["x"]) + float(part["w"]) / 2.0
    cy = float(part["y"]) + float(part["h"]) / 2.0
    sx, sy = canvas_to_spine(cx, cy)
    return sx - bone_world[0], sy - bone_world[1]


def classify_parts(parts: list[dict]) -> dict:
    hanging: list[str] = []
    sitting: list[str] = []
    baked_lights: list[str] = []
    for part in parts:
        if is_background(part):
            continue
        if is_baked_light(part):
            baked_lights.append(part["name"])
        elif is_hanging_lamp(part):
            hanging.append(part["name"])
        elif is_lamp(part):
            sitting.append(part["name"])
    extracted_lights = [f"{name}_light" for name in hanging + sitting]
    return {
        "hanging": hanging,
        "sitting": sitting,
        "baked_lights": baked_lights,
        "extracted_lights": extracted_lights,
        "flicker": extracted_lights + baked_lights,
        "lit": hanging + sitting,
    }


def pivot_kind(part: dict, hanging: set[str]) -> str:
    name = part["name"]
    if name in hanging:
        return "hang"
    if name == "hook":
        return "bottom"
    if name in {"beam", "background"} or is_background(part) or is_baked_light(part):
        return "center"
    return "bottom"


def pendulum_period(length_px: float) -> float:
    length = max(24.0, float(length_px))
    return 2.0 * math.pi * math.sqrt(length / GRAVITY_PX) * PENDULUM_TIME_SCALE


def mass_center(metrics: dict) -> tuple[float, float]:
    return float(metrics["center"][0]), float(metrics["center"][1])


def hang_length(metrics: dict, joint: tuple[float, float]) -> float:
    mx, my = mass_center(metrics)
    return float(math.hypot(mx - joint[0], my - joint[1]))


def diff_moved_layers(prev_path: Path, layers: list[dict]) -> list[dict]:
    if not prev_path.is_file():
        return []
    prev = json.loads(prev_path.read_text(encoding="utf-8"))
    old = {row["name"]: row.get("bounds") for row in prev.get("layers") or []}
    moved: list[dict] = []
    for row in layers:
        name = row.get("name")
        nb = row.get("bounds")
        ob = old.get(name)
        if not name or not nb or not ob:
            continue
        dx = float(nb[0]) - float(ob[0])
        dy = float(nb[1]) - float(ob[1])
        dw = (float(nb[2]) - float(nb[0])) - (float(ob[2]) - float(ob[0]))
        dh = (float(nb[3]) - float(nb[1])) - (float(ob[3]) - float(ob[1]))
        dist = math.hypot(dx, dy)
        if dist < 1.0 and abs(dw) < 1.0 and abs(dh) < 1.0:
            continue
        moved.append(
            {
                "name": name,
                "from": [round(float(ob[0]), 1), round(float(ob[1]), 1)],
                "to": [round(float(nb[0]), 1), round(float(nb[1]), 1)],
                "delta": [round(dx, 1), round(dy, 1)],
                "pixels": round(dist, 1),
                "size_delta": [round(dw, 1), round(dh, 1)],
            }
        )
    moved.sort(key=lambda m: m["pixels"], reverse=True)
    return moved


def load_psd2() -> dict:
    if not PSD2_PATH.is_file():
        raise SystemExit(f"missing {PSD2_PATH}; dump western_scene2.psd first")
    payload = json.loads(PSD2_PATH.read_text(encoding="utf-8"))
    exports = payload.get("exports")
    if not exports:
        raise SystemExit(f"{PSD2_PATH} has no exports; run crop_psd2_parts.py")
    canvas = payload.get("canvas") or {}
    global CANVAS_W, CANVAS_H
    CANVAS_W = int(canvas.get("w") or CANVAS_W)
    CANVAS_H = int(canvas.get("h") or CANVAS_H)
    return payload


def slot_draw_order(parts: list[dict], lit: list[str]) -> list[str]:
    # Spine draws first slot behind, last slot in front.
    # PSD dump is back-to-front (background first), so keep that order.
    lit_set = set(lit)
    back = [p for p in parts if is_background(p)]
    rest = [p for p in parts if not is_background(p)]
    rest.sort(key=lambda p: int(p.get("stack_index") or 0))
    names: list[str] = [p["name"] for p in back]
    for part in rest:
        names.append(part["name"])
        if part["name"] in lit_set:
            names.append(f"{part['name']}_light")
    return names


def prop_world(part: dict, metrics: dict, hanging: set[str]) -> dict:
    name = part["name"]
    scale = float(part["scale"])
    px, py = float(part["x"]), float(part["y"])
    native_w, native_h = int(part["native_w"]), int(part["native_h"])
    kind = pivot_kind(part, hanging)
    if kind == "hang":
        jx, jy = metrics["hang"]
    elif kind == "center":
        jx, jy = native_w / 2.0, native_h / 2.0
    else:
        jx, jy = metrics["bottom"]
    canvas_x = px + jx * scale
    canvas_y = py + jy * scale
    sx, sy = canvas_to_spine(canvas_x, canvas_y)
    ax, ay = attach_offset(native_w, native_h, jx, jy)
    return {
        "name": name,
        "filename": part["filename"],
        "psd_name": part.get("psd_name", name),
        "parent_group": part.get("parent_group", ""),
        "scale": scale,
        "pivot": kind,
        "joint_img": [jx, jy],
        "canvas_xy": [canvas_x, canvas_y],
        "world": [sx, sy],
        "attach": [ax, ay],
        "native": [native_w, native_h],
        "placed": [int(part["w"]), int(part["h"]), int(part["x"]), int(part["y"])],
    }


def copy_static_image(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def flicker_samples(index: int) -> list[tuple[float, float]]:
    patterns = [
        [
            (0.00, 0.85),
            (0.38, 1.00),
            (0.72, 0.58),
            (1.10, 0.94),
            (1.48, 0.70),
            (1.95, 1.00),
            (2.28, 0.55),
            (2.70, 0.92),
            (3.15, 0.68),
            (3.62, 0.98),
            (4.10, 0.62),
            (4.48, 0.90),
            (FLICKER_SRC_DUR, 0.85),
        ],
        [
            (0.00, 0.72),
            (0.28, 0.95),
            (0.62, 0.60),
            (1.05, 0.88),
            (1.40, 1.00),
            (1.78, 0.64),
            (2.25, 0.90),
            (2.68, 0.55),
            (3.10, 0.86),
            (3.55, 0.98),
            (4.00, 0.70),
            (4.42, 0.80),
            (FLICKER_SRC_DUR, 0.72),
        ],
        [
            (0.00, 0.88),
            (0.42, 1.00),
            (0.90, 0.82),
            (1.35, 0.97),
            (1.85, 0.86),
            (2.30, 1.00),
            (2.85, 0.84),
            (3.40, 0.95),
            (4.00, 0.88),
            (FLICKER_SRC_DUR, 0.88),
        ],
    ]
    return patterns[index % len(patterns)]


def sitting_flicker_samples() -> list[tuple[float, float]]:
    return [
        (0.00, 0.90),
        (0.38, 1.00),
        (0.82, 0.86),
        (1.28, 0.98),
        (1.74, 0.88),
        (2.22, 1.00),
        (2.70, 0.87),
        (3.20, 0.96),
        (3.72, 0.90),
        (4.20, 1.00),
        (FLICKER_SRC_DUR, 0.90),
    ]


def scale_flicker(samples: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    times = [float(t) for t, _ in samples]
    old = max(times) if times else FLICKER_SRC_DUR
    old = max(0.001, old)
    dur = max(0.001, float(duration))
    out = [(round(float(t) / old * dur, 4), a) for t, a in samples]
    if out:
        out[0] = (0.0, out[0][1])
        out[-1] = (round(dur, 4), out[0][1])
    return out


def build_clips(
    pendulums: list[dict],
    lights: list[str],
    duration: float,
    sitting: list[str] | None = None,
) -> dict:
    def white(a: float) -> str:
        return color_hex(1.0, 1.0, 1.0, a)

    sitting_lights = {f"{name}_light" for name in (sitting or [])}
    idle_bones: dict = {}
    for spec in pendulums:
        idle_bones[spec["name"]] = {
            "rotate": keyed_rotate_pendulum(spec["amplitude"], spec["period"], invert=spec["invert"])
        }

    idle_slots: dict = {}
    for i, slot in enumerate(lights):
        samples = sitting_flicker_samples() if slot in sitting_lights else flicker_samples(i)
        idle_slots[slot] = {
            "color": keyed_color(
                [(t, white(a)) for t, a in scale_flicker(samples, duration)]
            )
        }
    return {
        "idle": {"bones": idle_bones, "slots": idle_slots},
        "still": {"bones": {}, "slots": {}},
    }


def refine_hang_origins(
    parts: list[dict],
    metrics: dict[str, dict],
    worlds: dict[str, dict],
    hanging: list[str],
) -> dict[str, str]:
    """Keep hang origin on the chain rest (top-link centerline). Parent to hook if near."""
    parents = {name: "beam_anchor" for name in hanging}
    hook_part = next((p for p in parts if p["name"] == "hook"), None)
    hook_crotch = None
    if hook_part and "hook" in metrics:
        hook_m = metrics["hook"]
        hook_scale = float(hook_part.get("scale") or 1.0)
        hook_crotch = (
            float(hook_part["x"]) + float(hook_m["center"][0]) * hook_scale,
            float(hook_part["y"]) + float(hook_m["center"][1]) * hook_scale,
        )
    by_name = {p["name"]: p for p in parts}
    for name in hanging:
        part = by_name.get(name)
        if not part:
            continue
        hang = metrics[name]["hang"]
        jx, jy = float(hang[0]), float(hang[1])
        native_w, native_h = int(part["native_w"]), int(part["native_h"])
        scale = float(part.get("scale") or 1.0)
        canvas_x = float(part["x"]) + jx * scale
        canvas_y = float(part["y"]) + jy * scale
        sx, sy = canvas_to_spine(canvas_x, canvas_y)
        ax, ay = attach_offset(native_w, native_h, jx, jy)
        worlds[name]["joint_img"] = [jx, jy]
        worlds[name]["canvas_xy"] = [canvas_x, canvas_y]
        worlds[name]["world"] = [sx, sy]
        worlds[name]["attach"] = [ax, ay]
        worlds[name]["hang_source"] = "chain_rest"
        worlds[name]["pivot"] = "hang"
        if hook_crotch is None:
            continue
        if abs(hook_crotch[0] - canvas_x) <= HOOK_NEAR_X and abs(hook_crotch[1] - canvas_y) <= HOOK_NEAR_Y:
            parents[name] = "hook"
    return parents


def walk_curves(obj, path: str, errors: list[str]) -> None:
    if isinstance(obj, dict):
        curve = obj.get("curve")
        if isinstance(curve, list):
            errors.append(f"array curve at {path}")
        for k, v in obj.items():
            walk_curves(v, f"{path}.{k}", errors)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk_curves(v, f"{path}[{i}]", errors)


def last_angle(keys: list[dict] | None) -> float | None:
    if not keys:
        return None
    return float(keys[-1].get("angle", 0.0))


def first_angle(keys: list[dict] | None) -> float | None:
    if not keys:
        return None
    return float(keys[0].get("angle", 0.0))


def _placed_box(worlds: dict, name: str) -> tuple[float, float, float, float] | None:
    wld = worlds.get(name)
    if not wld:
        return None
    pw, ph, px, py = wld["placed"]
    return float(px), float(py), float(pw), float(ph)


def verify_scene(skeleton: dict, report_extra: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict] = []
    hanging = list(report_extra.get("hanging_lamps") or [])
    sitting = list(report_extra.get("sitting_lamps") or [])
    flicker = list(report_extra.get("light_slots") or [])
    extracted = set(report_extra.get("extracted_lights") or [])
    worlds = report_extra.get("pivots") or {}

    meta = skeleton["skeleton"]
    ok_ver = meta.get("spine") == SPINE_VERSION
    checks.append({"id": "spine_version", "ok": ok_ver, "got": meta.get("spine")})
    if not ok_ver:
        errors.append(f"spine version {meta.get('spine')} != {SPINE_VERSION}")

    size_ok = int(meta.get("width", 0)) == CANVAS_W and int(meta.get("height", 0)) == CANVAS_H
    checks.append({"id": "canvas", "ok": size_ok, "got": [meta.get("width"), meta.get("height")]})
    if not size_ok:
        errors.append(f"skeleton size {meta.get('width')}x{meta.get('height')} != {CANVAS_W}x{CANVAS_H}")

    images = meta.get("images") or ""
    img_ok = images.endswith("/") and not images.startswith("./")
    checks.append({"id": "images_abs", "ok": img_ok, "got": images})
    if not img_ok:
        errors.append("skeleton.images must be absolute and end with /")

    walk_curves(skeleton.get("animations", {}), "animations", errors)

    by_bone = {b["name"]: b for b in skeleton["bones"]}
    for name in hanging:
        bone = by_bone.get(name)
        if not bone:
            errors.append(f"missing hanging lamp bone {name}")
            continue
        rot = float(bone.get("rotation", 0.0))
        rot_ok = abs(rot) < 1e-6
        checks.append({"id": f"setup_rot:{name}", "ok": rot_ok, "rotation": rot})
        if not rot_ok:
            errors.append(f"{name} setup rotation={rot} (must be 0)")
        parent = bone.get("parent")
        parent_ok = parent in {"beam_anchor", "hook"}
        checks.append({"id": f"parent:{name}", "ok": parent_ok, "parent": parent})
        if not parent_ok:
            errors.append(f"{name} parent={parent} expected beam_anchor or hook")
        wld = worlds.get(name) or {}
        attach = wld.get("attach") or [0.0, 0.0]
        hang_down = float(attach[1]) < -20.0
        checks.append({"id": f"hang_down:{name}", "ok": hang_down, "attach": attach})
        if not hang_down:
            errors.append(f"{name} attachment does not hang below origin (attach={attach})")
        chain_x = abs(float(attach[0])) <= 8.0
        checks.append({"id": f"chain_center:{name}", "ok": chain_x, "attach": attach})
        if not chain_x:
            errors.append(f"{name} pivot is off the chain centerline (attach x={attach[0]})")

    idle = skeleton["animations"].get("idle", {})
    for name in hanging:
        keys = idle.get("bones", {}).get(name, {}).get("rotate")
        a0 = first_angle(keys)
        a1 = last_angle(keys)
        loop_ok = (
            a0 is not None
            and a1 is not None
            and abs(a0 - a1) < 0.05
            and abs(a0) >= 5.0
        )
        checks.append({"id": f"idle_loop:{name}", "ok": loop_ok, "first": a0, "last": a1})
        if not loop_ok:
            errors.append(f"idle.{name} pendulum does not seal at extreme (first={a0} last={a1})")
        tr = idle.get("bones", {}).get(name, {}).get("translate")
        if tr:
            errors.append(f"idle.{name} has translate keys (pendulum is rotate-only)")
        last_t = float(keys[-1].get("time", 0.0)) if keys else 0.0
        want_t = next((float(spec["period"]) for spec in (report_extra.get("pendulum") or []) if spec.get("name") == name), None)
        if want_t is not None and keys and abs(last_t - want_t) > 0.02:
            errors.append(f"idle.{name} last key {last_t} != period {want_t}")
    periods = [float(spec["period"]) for spec in (report_extra.get("pendulum") or [])]
    async_ok = len(periods) < 2 or max(periods) - min(periods) > 0.02
    checks.append({"id": "pendulum_async", "ok": async_ok, "periods": periods})
    if not async_ok:
        errors.append(f"hanging lamps share a period {periods}; they must drift")

    def _max_key_time(obj) -> float:
        found = 0.0
        if isinstance(obj, dict):
            if "time" in obj:
                found = max(found, float(obj.get("time") or 0.0))
            for v in obj.values():
                found = max(found, _max_key_time(v))
        elif isinstance(obj, list):
            for v in obj:
                found = max(found, _max_key_time(v))
        return found

    idle_max = _max_key_time(idle)
    hold_ok = idle_max <= IDLE_DURATION + 0.02
    checks.append({"id": "no_trailing_hold", "ok": hold_ok, "max_key": idle_max, "duration": IDLE_DURATION})
    if not hold_ok:
        errors.append(f"idle has keys past clip duration ({idle_max} > {IDLE_DURATION}); lamp will rest")

    slots = {s["name"]: s for s in skeleton["slots"]}
    for lamp in hanging + sitting:
        body = slots.get(lamp)
        if not body:
            errors.append(f"missing body slot {lamp}")
            continue
        if body.get("color") not in (None, "ffffffff"):
            warnings.append(f"{lamp} body slot has non-white setup color")
        body_keys = idle.get("slots", {}).get(lamp, {}).get("color")
        if body_keys:
            errors.append(f"idle flickers body slot {lamp} (must be light only)")

    for light_name in flicker:
        light = slots.get(light_name)
        if not light:
            errors.append(f"missing light slot {light_name}")
            continue
        light_keys = idle.get("slots", {}).get(light_name, {}).get("color")
        if not light_keys:
            errors.append(f"idle missing color keys on {light_name}")
        if light_name in extracted:
            blend_ok = light.get("blend") == "additive"
            checks.append({"id": f"blend:{light_name}", "ok": blend_ok})
            if not blend_ok:
                errors.append(f"{light_name} blend is not additive")

    slot_names = [s["name"] for s in skeleton["slots"]]
    bg_first = bool(slot_names) and is_background({"name": slot_names[0]})
    checks.append({"id": "background_behind", "ok": bg_first, "first": slot_names[0] if slot_names else None})
    if not bg_first:
        errors.append(f"background must be the first slot (behind); first={slot_names[0] if slot_names else None}")
    for banned in ("ground_mist", "ground_fire"):
        present = banned in slots
        checks.append({"id": f"no_{banned}", "ok": not present})
        if present:
            errors.append(f"{banned} must not be in the scene")

    world = resolve_world_bones(skeleton["bones"])
    for name in hanging + ["root", "beam_anchor", "bg"]:
        if name not in world:
            errors.append(f"unresolved bone {name}")

    sign = _placed_box(worlds, "signpost")
    if sign:
        left_ok = sign[0] < CANVAS_W * 0.28
        checks.append({"id": "signpost_left", "ok": left_ok, "box": list(sign)})
        if not left_ok:
            errors.append(f"signpost not on the left: {sign}")
    barrel = _placed_box(worlds, "barrel")
    if barrel:
        right_ok = barrel[0] > CANVAS_W * 0.55
        checks.append({"id": "barrel_right", "ok": right_ok, "box": list(barrel)})
        if not right_ok:
            errors.append(f"barrel not on the right: {barrel}")
    beam = _placed_box(worlds, "beam")
    post_l = _placed_box(worlds, "post_L")
    post_r = _placed_box(worlds, "post_R")
    if beam and post_l and post_r:
        beam_l, _, beam_w, _ = beam
        beam_r = beam_l + beam_w
        covers = beam_l <= post_l[0] + post_l[2] * 0.35 and beam_r >= post_r[0] + post_r[2] * 0.65
        checks.append({"id": "beam_spans_posts", "ok": covers, "beam": list(beam)})
        if not covers:
            errors.append(f"beam {beam} does not span posts {post_l} {post_r}")

    if not hanging:
        errors.append("no hanging lamps classified from PSD")
    if not flicker:
        errors.append("no light slots to flicker")
    if "light" in hanging:
        errors.append("baked PSD light layer was classified as a hanging lamp")
    for name in hanging:
        if name in LIGHT_TOKENS:
            errors.append(f"{name} must not sway; it is a light wash")
    for name in report_extra.get("baked_lights") or []:
        bone = by_bone.get(name)
        if bone and bone.get("parent") == "beam_anchor":
            errors.append(f"baked light {name} parented to beam_anchor")
        if idle.get("bones", {}).get(name, {}).get("rotate"):
            errors.append(f"baked light {name} has idle rotate")

    for word in SIGN_WORDS:
        slot = slots.get(word["slot"])
        if not slot:
            errors.append(f"missing sign word slot {word['slot']}")
            continue
        if slot.get("bone") != "signpost":
            errors.append(f"{word['slot']} bone={slot.get('bone')} expected signpost")
    hi = slots.get(SIGN_HIGHLIGHT_SLOT)
    if not hi:
        errors.append("missing sign_highlight slot")
    elif hi.get("bone") != "signpost":
        errors.append(f"sign_highlight bone={hi.get('bone')} expected signpost")
    if idle.get("slots", {}).get(SIGN_HIGHLIGHT_SLOT):
        errors.append("idle keys sign_highlight; highlight is derived, not animated")
    left = next((p for p in (report_extra.get("pendulum") or []) if p["name"] == "left_hanging_lamp"), None)
    if left and left.get("hang_source") != "chain_rest":
        errors.append(f"left hang_source={left.get('hang_source')} expected chain_rest")

    ok = len(errors) == 0
    return {
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "tol_px": 1.5,
        **report_extra,
    }


def write_placement(payload: dict, parts: list[dict], classified: dict) -> None:
    out = {
        "source": "western_scene2.psd",
        "source_full": payload.get("source_full"),
        "method": "photoshop_bounds",
        "canvas": {"w": CANVAS_W, "h": CANVAS_H},
        "part_count": len(parts),
        "hanging_lamps": classified["hanging"],
        "sitting_lamps": classified["sitting"],
        "baked_lights": classified["baked_lights"],
        "parts": [
            {
                "filename": p["filename"],
                "name": p["name"],
                "psd_name": p.get("psd_name"),
                "parent_group": p.get("parent_group"),
                "x": p["x"],
                "y": p["y"],
                "w": p["w"],
                "h": p["h"],
                "scale": p["scale"],
                "native_w": p["native_w"],
                "native_h": p["native_h"],
                "stack_index": p.get("stack_index"),
                "source": "photoshop_bounds",
            }
            for p in parts
        ],
    }
    PLACEMENT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")


def composite_check(parts: list[dict], dest: Path) -> None:
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 255))
    ordered = sorted(parts, key=lambda p: int(p.get("stack_index") or 0), reverse=True)
    for part in ordered:
        src = PARTS_DIR / part["filename"]
        if not src.is_file():
            continue
        im = Image.open(src).convert("RGBA")
        if is_background(part):
            canvas = Image.alpha_composite(canvas, im)
            continue
        tile = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
        tile.paste(im, (int(part["x"]), int(part["y"])), im)
        canvas = Image.alpha_composite(canvas, tile)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(dest)


def build() -> dict:
    payload = load_psd2()
    parts = apply_sign_names([p for p in payload["exports"] if p.get("name")])
    if not any(is_background(p) for p in parts):
        raise SystemExit("psd2 exports missing background")
    classified = classify_parts(parts)
    hanging = classified["hanging"]
    hanging_set = set(hanging)
    sitting_set = set(classified["sitting"])
    lit_set = hanging_set | sitting_set
    write_placement(payload, parts, classified)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
    IMAGES_DIR.mkdir(parents=True)

    metrics: dict[str, dict] = {}
    worlds: dict[str, dict] = {}
    glow_info: dict[str, dict] = {}

    for part in parts:
        name = part["name"]
        src = PARTS_DIR / part["filename"]
        if not src.is_file():
            raise SystemExit(f"missing part {src}")
        with Image.open(src) as im:
            arr = np.asarray(im.convert("RGBA"))
            native = (int(part["native_w"]), int(part["native_h"]))
            if im.size != native:
                raise SystemExit(f"{name} PNG {im.size} != native {native}")
        metrics[name] = alpha_metrics(arr)
        worlds[name] = prop_world(part, metrics[name], hanging_set)
        dest = IMAGES_DIR / f"{name}.png"
        if name in lit_set:
            glow_info[name] = extract_lamp_layers(
                src,
                dest,
                IMAGES_DIR / f"{name}_light.png",
                force_on=name in sitting_set,
            )
        else:
            copy_static_image(src, dest)

    hang_parents = refine_hang_origins(parts, metrics, worlds, hanging)
    moved = diff_moved_layers(PREV_JSON, payload.get("layers") or [])

    pendulums: list[dict] = []
    lengths: list[float] = []
    for i, name in enumerate(hanging):
        joint = tuple(worlds[name]["joint_img"])
        length = hang_length(metrics[name], joint) * float(worlds[name]["scale"])
        lengths.append(length)
        period = pendulum_period(length) * PENDULUM_PERIOD_SCALE[i % len(PENDULUM_PERIOD_SCALE)]
        amp = PENDULUM_AMP[i % len(PENDULUM_AMP)]
        pendulums.append(
            {
                "name": name,
                "length_px": round(length, 2),
                "period": round(period, 4),
                "amplitude": amp,
                "invert": i % 2 == 1,
                "parent": hang_parents[name],
                "hang_origin_canvas": [round(worlds[name]["canvas_xy"][0], 2), round(worlds[name]["canvas_xy"][1], 2)],
                "hang_origin_spine": [round(worlds[name]["world"][0], 2), round(worlds[name]["world"][1], 2)],
                "hang_source": worlds[name].get("hang_source", "chain_top"),
                "attach": [round(worlds[name]["attach"][0], 3), round(worlds[name]["attach"][1], 3)],
            }
        )
    global IDLE_DURATION
    IDLE_DURATION = max((spec["period"] for spec in pendulums), default=2.4)

    if "beam" not in worlds:
        raise SystemExit("PSD has no beam layer; cannot parent hanging lamps")

    beam_w = worlds["beam"]["world"]
    bones: list[dict] = [
        {"name": "root", "x": 0.0, "y": 0.0},
        {"name": "bg", "parent": "root", "x": CANVAS_W / 2.0, "y": CANVAS_H / 2.0},
        {
            "name": "beam_anchor",
            "parent": "root",
            "x": round(beam_w[0], 3),
            "y": round(beam_w[1], 3),
        },
    ]

    def add_bone(
        name: str,
        parent: str,
        world_xy: tuple[float, float],
        scale: float,
        length: float | None = None,
    ) -> None:
        if parent == "root":
            parent_xy = (0.0, 0.0)
        else:
            resolved = resolve_world_bones(bones)
            parent_xy = (resolved[parent][0], resolved[parent][1])
        rec = {
            "name": name,
            "parent": parent,
            "x": round(world_xy[0] - parent_xy[0], 3),
            "y": round(world_xy[1] - parent_xy[1], 3),
        }
        if abs(scale - 1.0) > 1e-6:
            rec["scaleX"] = round(scale, 4)
            rec["scaleY"] = round(scale, 4)
        if length is not None:
            rec["length"] = length
        bones.append(rec)

    add_bone("beam", "beam_anchor", beam_w, worlds["beam"]["scale"], 80.0)
    if "hook" in worlds and "hook" in hang_parents.values():
        add_bone("hook", "beam_anchor", tuple(worlds["hook"]["world"]), worlds["hook"]["scale"], 16.0)
    by_pendulum = {p["name"]: p for p in pendulums}
    for name in hanging:
        parent = hang_parents.get(name, "beam_anchor")
        length = by_pendulum[name]["length_px"] if name in by_pendulum else 48.0
        add_bone(name, parent, tuple(worlds[name]["world"]), worlds[name]["scale"], length)
        add_bone(f"{name}_light", name, tuple(worlds[name]["world"]), 1.0, 18.0)

    for name in classified["sitting"]:
        add_bone(name, "root", tuple(worlds[name]["world"]), worlds[name]["scale"], 36.0)
        add_bone(f"{name}_light", name, tuple(worlds[name]["world"]), 1.0, 18.0)
    for name in classified["baked_lights"]:
        parent = classified["sitting"][0] if classified["sitting"] else "root"
        add_bone(name, parent, tuple(worlds[name]["world"]), worlds[name]["scale"], 16.0)

    overlay_names = {p["name"] for p in parts if is_sign_overlay(p)}
    bg_names = {p["name"] for p in parts if is_background(p)}
    already = {"background", "beam"} | bg_names | hanging_set | set(classified["sitting"]) | set(classified["baked_lights"]) | overlay_names
    if "hook" in hang_parents.values():
        already.add("hook")
    draw_names = slot_draw_order(parts, classified["lit"])
    for name in draw_names:
        if name.endswith("_light") or name in already:
            continue
        add_bone(name, "root", tuple(worlds[name]["world"]), worlds[name]["scale"], 24.0)

    slots: list[dict] = []
    default_skin: dict = {}
    attachments_meta: dict = {}

    def add_slot(
        slot_name: str,
        bone_name: str,
        att_name: str,
        attach: tuple[float, float],
        size: tuple[int, int],
        *,
        additive: bool = False,
        scale_xy: tuple[float, float] | None = None,
    ) -> None:
        rec = {"name": slot_name, "bone": bone_name, "attachment": att_name}
        if additive:
            rec["blend"] = "additive"
            rec["color"] = "ffffffee"
        slots.append(rec)
        att = {
            "x": round(attach[0], 3),
            "y": round(attach[1], 3),
            "width": int(size[0]),
            "height": int(size[1]),
        }
        if scale_xy is not None:
            sx, sy = float(scale_xy[0]), float(scale_xy[1])
            if abs(sx - 1.0) > 1e-6:
                att["scaleX"] = round(sx, 4)
            if abs(sy - 1.0) > 1e-6:
                att["scaleY"] = round(sy, 4)
        default_skin.setdefault(slot_name, {})[att_name] = att
        attachments_meta[slot_name] = att

    for slot_name in draw_names:
        if is_background({"name": slot_name}) or slot_name == "background":
            add_slot(slot_name, "bg", slot_name, (0.0, 0.0), (CANVAS_W, CANVAS_H))
            continue
        if slot_name.endswith("_light") and slot_name in classified["extracted_lights"]:
            lamp = slot_name[: -len("_light")]
            wld = worlds[lamp]
            info = glow_info.get(lamp) or {}
            if lamp in sitting_set:
                part = next(p for p in parts if p["name"] == lamp)
                flame = info.get("flame_img") or [float(part["w"]) * 0.5, float(part["h"]) * 0.52]
                fx = float(part["x"]) + float(flame[0]) * float(part["scale"])
                fy = float(part["y"]) + float(flame[1]) * float(part["scale"])
                flame_world = canvas_to_spine(fx, fy)
                bone_xy = tuple(wld["world"])
                attach = (flame_world[0] - bone_xy[0], flame_world[1] - bone_xy[1])
                size = tuple(info.get("glow_size") or [SITTING_GLOW_SIZE, SITTING_GLOW_SIZE])
            else:
                attach = tuple(wld["attach"])
                size = tuple(wld["native"])
            add_slot(
                slot_name,
                slot_name,
                slot_name,
                attach,
                size,
                additive=True,
            )
            continue
        if slot_name in overlay_names:
            if "signpost" not in worlds:
                raise SystemExit("sign overlays need a signpost bone")
            bone_xy = tuple(worlds["signpost"]["world"])
            part = next(p for p in parts if p["name"] == slot_name)
            attach = overlay_attach_on_bone(part, bone_xy)
            native = worlds[slot_name]["native"]
            if int(part["w"]) != native[0] or int(part["h"]) != native[1]:
                raise SystemExit(
                    f"{slot_name} stretch: canvas {part['w']}x{part['h']} != png {native[0]}x{native[1]}"
                )
            if abs(float(part.get("scale") or 1.0) - 1.0) > 1e-6:
                raise SystemExit(f"{slot_name} has scale {part['scale']}; overlays must be 1:1")
            add_slot(slot_name, "signpost", slot_name, attach, tuple(native))
            continue
        wld = worlds[slot_name]
        add_slot(slot_name, slot_name, slot_name, tuple(wld["attach"]), tuple(wld["native"]))

    animations = build_clips(
        pendulums,
        classified["flicker"],
        IDLE_DURATION,
        sitting=classified["sitting"],
    )
    sign_info = default_sign_state()
    lamp_info = default_lamp_state(classified["sitting"])
    SIGN_STATE_PATH.write_text(json.dumps(sign_info, indent=2), encoding="utf-8")
    LAMP_STATE_PATH.write_text(json.dumps(lamp_info, indent=2), encoding="utf-8")
    (OUT_DIR / "sign_state.json").write_text(json.dumps(sign_info, indent=2), encoding="utf-8")
    (OUT_DIR / "lamp_state.json").write_text(json.dumps(lamp_info, indent=2), encoding="utf-8")

    images_abs = str(IMAGES_DIR.resolve()).replace("\\", "/")
    if not images_abs.endswith("/"):
        images_abs += "/"

    skeleton = {
        "skeleton": {
            "hash": "western-scene2",
            "spine": SPINE_VERSION,
            "x": 0,
            "y": 0,
            "width": CANVAS_W,
            "height": CANVAS_H,
            "images": images_abs,
            "audio": "",
        },
        "bones": bones,
        "slots": slots,
        "skins": {"default": default_skin},
        "animations": animations,
    }

    extra = {
        "source_psd": payload.get("saved_copy") or payload.get("source_full"),
        "source_open": payload.get("source_open"),
        "method": "photoshop_bounds",
        "canvas": {"w": CANVAS_W, "h": CANVAS_H},
        "bones": [b["name"] for b in bones],
        "slots": [s["name"] for s in slots],
        "clips": {
            "idle": {"duration": IDLE_DURATION, "loop": True, "use": "pendulum + flicker"},
            "still": {"duration": 0.0, "loop": True, "use": "A/B posed scene"},
        },
        "moved_layers": moved,
        "pendulum": pendulums,
        "gravity_px": GRAVITY_PX,
        "hanging_lamps": hanging,
        "sitting_lamps": classified["sitting"],
        "baked_lights": classified["baked_lights"],
        "extracted_lights": classified["extracted_lights"],
        "light_slots": classified["flicker"],
        "glow": glow_info,
        "lamps": lamp_info,
        "pivots": {n: worlds[n] for n in worlds},
        "layer_count_psd": int(payload.get("layer_count") or 0),
        "prop_count": len(parts),
        "serve": "python -m http.server 8780 --bind 127.0.0.1",
        "serve_dir": str(OUT_DIR),
        "viewer_url": "http://127.0.0.1:8780/viewer.html",
        "sign": sign_info,
    }
    report = verify_scene(skeleton, extra)
    (OUT_DIR / "skeleton.json").write_text(json.dumps(skeleton, indent=2), encoding="utf-8")
    (OUT_DIR / "western-scene.json").write_text(json.dumps(skeleton, indent=2), encoding="utf-8")
    presets = {
        "title": "western scene 2",
        "clips": [
            {"id": "idle", "title": "idle", "use": "pendulum + flicker", "loop": True},
            {"id": "still", "title": "still", "use": "no motion", "loop": True},
        ],
    }
    (OUT_DIR / "presets.json").write_text(json.dumps(presets, indent=2), encoding="utf-8")
    (OUT_DIR / "SETUP_POSE_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "scene_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if VIEWER_SRC.is_file():
        shutil.copy2(VIEWER_SRC, OUT_DIR / "viewer.html")
    composite_check(parts, OUT_DIR / "psd2_composite.png")
    if not report["ok"]:
        raise SystemExit("verify_scene FAILED:\n  - " + "\n  - ".join(report["errors"]))
    return report


def main() -> int:
    report = build()
    print(f"wrote {OUT_DIR / 'skeleton.json'}")
    print(f"images {IMAGES_DIR}")
    print(f"canvas {report['canvas']}")
    print(f"clips {list(report['clips'])}")
    print(f"hanging {report['hanging_lamps']}")
    sign = report.get("sign") or {}
    print(f"sign words {[(w['slot'], sign.get(w['id'])) for w in SIGN_WORDS]} highlight={sign.get('highlight')}")
    print(f"sitting {report['sitting_lamps']}")
    print(f"baked_lights {report['baked_lights']}")
    print(f"lights {report['light_slots']}")
    print(f"lamps {report.get('lamps')}")
    print(f"moved {len(report.get('moved_layers') or [])}")
    for row in report.get("moved_layers") or []:
        print(f"  {row['name']:22s} {row['from']} -> {row['to']}  d=({row['delta'][0]:+.1f},{row['delta'][1]:+.1f})  {row['pixels']}px")
    for spec in report.get("pendulum") or []:
        print(
            f"  pendulum {spec['name']}: L={spec['length_px']}px T={spec['period']}s "
            f"A={spec['amplitude']} invert={spec['invert']} parent={spec['parent']} "
            f"hang={spec['hang_source']} origin={spec['hang_origin_canvas']} attach={spec['attach']}"
        )
    for name, info in report["glow"].items():
        print(f"  glow {name}: {info['method']} core={info['core_px']} glow_px={info['glow_opaque']}")
    print("ok", report["ok"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
