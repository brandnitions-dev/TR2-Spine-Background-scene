"""
Match split prop PNGs onto the reference scene and write a layered PSD.

Re-run:
  python western-props/split_sheet.py backgroundSPINE/props.png -o backgroundSPINE/parts --no-keyed
  python backgroundSPINE/build_western_psd.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARTS_DIR = HERE / "parts"
REF_PATH = HERE / "backgroundimagewith props_ref.png"
BG_PATH = HERE / "backgroudn image no props.png"
PLACEMENT_PATH = HERE / "placement.json"
PREVIEW_PATH = HERE / "western_scene_preview.png"
PSD_PATH = HERE / "western_scene.psd"

SCALES_LARGE = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0)
SCALES_MED = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85, 0.95, 1.0)
SCALES_SMALL = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0)
WEAK_SCORE = 0.35
OVERLAP_IOU = 0.28
RESIDUAL_DELTA = 16
RESID_WEIGHT = 0.45
PEAKS_PER_SCALE = 3
SHEET_NAMES = {
    "prop_01.png": "post",
    "prop_02.png": "post",
    "prop_03.png": "beam",
    "prop_04.png": "lantern_hang",
    "prop_05.png": "lantern_lit",
    "prop_06.png": "chain",
    "prop_07.png": "chain",
    "prop_08.png": "signpost",
    "prop_09.png": "lantern_dim",
    "prop_10.png": "hook",
    "prop_11.png": "tombstone",
    "prop_12.png": "wagon_wheel",
    "prop_13.png": "barrel",
    "prop_14.png": "skull",
    "prop_15.png": "grass",
    "prop_16.png": "cowboy_hat",
    "prop_17.png": "grass",
    "prop_18.png": "shrub",
    "prop_19.png": "rocks",
    "prop_20.png": "grass",
    "prop_21.png": "rock",
    "prop_22.png": "rock",
    "prop_23.png": "rocks",
    "prop_24.png": "rock",
}


def load_rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def residual_mask(ref_rgb: np.ndarray, bg_rgb: np.ndarray) -> np.ndarray:
    delta = np.abs(ref_rgb.astype(np.int16) - bg_rgb.astype(np.int16)).mean(axis=2)
    return delta >= RESIDUAL_DELTA


def scales_for(rec: dict) -> tuple[float, ...]:
    w, h = rec["width"], rec["height"]
    if h > 400 or w > 400:
        return SCALES_LARGE
    if max(w, h) < 140:
        return SCALES_SMALL
    return SCALES_MED


def residual_overlap(resid: np.ndarray, alpha: np.ndarray, x: int, y: int) -> float:
    ys, xs = np.where(alpha > 12)
    if ys.size == 0:
        return 0.0
    yy = y + ys
    xx = x + xs
    valid = (yy >= 0) & (yy < resid.shape[0]) & (xx >= 0) & (xx < resid.shape[1])
    if not valid.any():
        return 0.0
    return float(resid[yy[valid], xx[valid]].mean())


def collect_candidates(
    scene_bgr: np.ndarray,
    resid: np.ndarray,
    templ_bgra: np.ndarray,
    scales: tuple[float, ...],
) -> list[dict]:
    th0, tw0 = templ_bgra.shape[:2]
    sh, sw = scene_bgr.shape[:2]
    templ_bgr = cv2.cvtColor(templ_bgra[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha0 = templ_bgra[:, :, 3]
    found: list[dict] = []
    for scale in scales:
        tw = max(8, int(round(tw0 * scale)))
        th = max(8, int(round(th0 * scale)))
        if tw >= sw or th >= sh:
            continue
        if abs(scale - 1.0) < 1e-6:
            t = templ_bgr
            m = alpha0
        else:
            t = cv2.resize(templ_bgr, (tw, th), interpolation=cv2.INTER_AREA)
            m = cv2.resize(alpha0, (tw, th), interpolation=cv2.INTER_AREA)
        if int((m > 12).sum()) < 40:
            continue
        try:
            res = cv2.matchTemplate(scene_bgr, t, cv2.TM_CCOEFF_NORMED, mask=m)
        except cv2.error:
            continue
        work = np.where(np.isfinite(res), res, -1e9).astype(np.float32)
        for _ in range(PEAKS_PER_SCALE):
            _, max_val, _, max_loc = cv2.minMaxLoc(work)
            score = float(max_val)
            if not np.isfinite(score) or score < 0.05:
                break
            x, y = int(max_loc[0]), int(max_loc[1])
            ov = residual_overlap(resid, m, x, y)
            found.append(
                {
                    "ccoeff": score,
                    "resid_ov": ov,
                    "score": score + RESID_WEIGHT * ov,
                    "x": x,
                    "y": y,
                    "scale": float(scale),
                    "w": tw,
                    "h": th,
                    "source": "ccoeff+residual",
                }
            )
            y0 = max(0, y - th // 2)
            x0 = max(0, x - tw // 2)
            y1 = min(work.shape[0], y + th // 2)
            x1 = min(work.shape[1], x + tw // 2)
            work[y0:y1, x0:x1] = -1e9
    found.sort(key=lambda c: c["score"], reverse=True)
    return found


def iou(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union else 0.0


def semantic_name(filename: str, place: dict, used: dict[str, int]) -> str:
    base = SHEET_NAMES.get(filename, Path(filename).stem)
    if base == "post":
        cx = place["x"] + place["w"] / 2
        base = "post_L" if cx < 768 else "post_R"
    n = used.get(base, 0) + 1
    used[base] = n
    return base if n == 1 else f"{base}_{n:02d}"


def pick_candidate(cands: list[dict], placed: list[dict], rec: dict) -> dict:
    if not cands:
        return {
            "ccoeff": 0.0,
            "resid_ov": 0.0,
            "score": 0.0,
            "x": 0,
            "y": 0,
            "scale": 1.0,
            "w": rec["width"],
            "h": rec["height"],
            "source": "fallback_origin",
        }
    name = SHEET_NAMES.get(rec["filename"], "")
    need_side = None
    if name == "post":
        have_l = any(p["name"].startswith("post_L") for p in placed)
        have_r = any(p["name"].startswith("post_R") for p in placed)
        if have_l and not have_r:
            need_side = "R"
        elif have_r and not have_l:
            need_side = "L"
    for cand in cands:
        if any(iou(cand, p) > OVERLAP_IOU for p in placed):
            continue
        cx = cand["x"] + cand["w"] / 2
        if need_side == "L" and cx >= 768:
            continue
        if need_side == "R" and cx < 768:
            continue
        return cand
    for cand in cands:
        if any(iou(cand, p) > OVERLAP_IOU for p in placed):
            continue
        return cand
    return cands[0]


def match_all(parts: list[dict]) -> list[dict]:
    ref_rgb = load_rgb(REF_PATH)
    bg_rgb = load_rgb(BG_PATH)
    ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)
    resid = residual_mask(ref_rgb, bg_rgb)
    sh, sw = ref_rgb.shape[:2]
    print(f"scene {sw}x{sh} residual_frac={float(resid.mean()):.3f}")

    raw: list[dict] = []
    for rec in parts:
        templ = load_rgba(PARTS_DIR / rec["filename"])
        cands = collect_candidates(ref_bgr, resid, templ, scales_for(rec))
        raw.append({"rec": rec, "cands": cands})
        top = cands[0] if cands else None
        if top:
            print(
                f"  match {rec['filename']:14s}  {top['w']}x{top['h']}  "
                f"xy=({top['x']},{top['y']})  scale={top['scale']:.2f}  "
                f"ccoeff={top['ccoeff']:.3f}  resid={top['resid_ov']:.3f}  "
                f"score={top['score']:.3f}"
            )
        else:
            print(f"  match {rec['filename']:14s}  NO CANDIDATE")

    raw.sort(key=lambda r: r["cands"][0]["score"] if r["cands"] else -1, reverse=True)
    placed: list[dict] = []
    used_names: dict[str, int] = {}
    for item in raw:
        rec = item["rec"]
        chosen = pick_candidate(item["cands"], placed, rec)
        name = semantic_name(rec["filename"], chosen, used_names)
        entry = {
            "filename": rec["filename"],
            "name": name,
            "x": int(chosen["x"]),
            "y": int(chosen["y"]),
            "w": int(chosen["w"]),
            "h": int(chosen["h"]),
            "scale": float(chosen["scale"]),
            "score": float(chosen["score"]),
            "ccoeff": float(chosen.get("ccoeff", chosen["score"])),
            "resid_ov": float(chosen.get("resid_ov", 0.0)),
            "source": chosen.get("source", "ccoeff+residual"),
            "sheet_bbox": rec["bbox"],
            "sheet_area": rec["area"],
            "native_w": rec["width"],
            "native_h": rec["height"],
        }
        placed.append(entry)

    refine_structure(placed, ref_bgr, resid)
    placed.sort(key=layer_order)
    return placed


def layer_order(p: dict) -> tuple:
    name = p["name"]
    if name.startswith("post") or name == "beam":
        return (1, p["y"], p["x"])
    if name in {"signpost", "tombstone", "barrel", "wagon_wheel"} or name.startswith("chain"):
        return (2, p["y"] + p["h"], p["x"])
    if "lantern" in name or name == "hook":
        return (3, p["y"], p["x"])
    return (4, p["y"] + p["h"], p["x"])


def _apply_cand(entry: dict, cand: dict) -> None:
    entry["x"] = int(cand["x"])
    entry["y"] = int(cand["y"])
    entry["w"] = int(cand["w"])
    entry["h"] = int(cand["h"])
    entry["scale"] = float(cand["scale"])
    entry["score"] = float(cand["score"])
    entry["ccoeff"] = float(cand.get("ccoeff", cand["score"]))
    entry["resid_ov"] = float(cand.get("resid_ov", 0.0))
    entry["source"] = cand.get("source", "ccoeff+residual") + "+structure"


def refine_structure(placed: list[dict], ref_bgr: np.ndarray, resid: np.ndarray) -> None:
    """Snap beam across both posts; keep post scales consistent."""
    by_name = {p["name"]: p for p in placed}
    left = by_name.get("post_L")
    right = by_name.get("post_R")
    beam = by_name.get("beam")
    if left and right:
        target_scale = left["scale"]
        right_rec = {
            "filename": right["filename"],
            "width": right["native_w"],
            "height": right["native_h"],
        }
        templ = load_rgba(PARTS_DIR / right["filename"])
        cands = collect_candidates(ref_bgr, resid, templ, SCALES_LARGE)
        best = None
        best_key = -1e9
        for cand in cands:
            cx = cand["x"] + cand["w"] / 2
            if cx < 800:
                continue
            key = cand["score"] - 1.6 * abs(cand["scale"] - target_scale)
            if key > best_key:
                best_key = key
                best = cand
        if best is not None:
            _apply_cand(right, best)
            print(
                f"  refine post_R -> ({right['x']},{right['y']}) "
                f"{right['w']}x{right['h']} s={right['scale']:.2f}"
            )

    left = by_name.get("post_L")
    right = by_name.get("post_R")
    if left and right and beam:
        templ = load_rgba(PARTS_DIR / beam["filename"])
        cands = collect_candidates(ref_bgr, resid, templ, SCALES_LARGE)
        top_y = min(left["y"], right["y"])
        need_l = left["x"] + left["w"] * 0.35
        need_r = right["x"] + right["w"] * 0.65
        best = None
        best_key = -1e9
        for cand in cands:
            covers = cand["x"] <= need_l and (cand["x"] + cand["w"]) >= need_r
            y_pen = abs(cand["y"] - (top_y + 25)) / 80.0
            key = cand["score"] + (0.55 if covers else -0.25) - 0.15 * y_pen
            if key > best_key:
                best_key = key
                best = cand
        if best is not None:
            _apply_cand(beam, best)
            print(
                f"  refine beam -> ({beam['x']},{beam['y']}) "
                f"{beam['w']}x{beam['h']} s={beam['scale']:.2f} "
                f"span={beam['x']}-{beam['x']+beam['w']}"
            )


def paste_clip(canvas: Image.Image, im: Image.Image, x: int, y: int) -> None:
    cw, ch = canvas.size
    iw, ih = im.size
    sx0 = max(0, -x)
    sy0 = max(0, -y)
    dx0 = max(0, x)
    dy0 = max(0, y)
    sx1 = min(iw, cw - dx0 + sx0)
    sy1 = min(ih, ch - dy0 + sy0)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    tile = im.crop((sx0, sy0, sx1, sy1))
    canvas.alpha_composite(tile, (dx0, dy0))


def composite_preview(placed: list[dict], canvas_size: tuple[int, int]) -> Image.Image:
    bg = Image.open(BG_PATH).convert("RGBA")
    if bg.size != canvas_size:
        bg = bg.resize(canvas_size, Image.Resampling.LANCZOS)
    canvas = bg
    for p in placed:
        im = Image.open(PARTS_DIR / p["filename"]).convert("RGBA")
        if abs(p["scale"] - 1.0) > 0.001:
            im = im.resize((p["w"], p["h"]), Image.Resampling.LANCZOS)
        paste_clip(canvas, im, int(p["x"]), int(p["y"]))
    return canvas


def write_psd_tools(placed: list[dict], canvas_size: tuple[int, int]) -> None:
    psd = PSDImage.new("RGB", canvas_size, color=(0, 0, 0))
    bg = Image.open(BG_PATH).convert("RGBA")
    if bg.size != canvas_size:
        bg = bg.resize(canvas_size, Image.Resampling.LANCZOS)
    psd.create_pixel_layer(bg, name="background", top=0, left=0)
    for p in placed:
        im = Image.open(PARTS_DIR / p["filename"]).convert("RGBA")
        if abs(p["scale"] - 1.0) > 0.001:
            im = im.resize((p["w"], p["h"]), Image.Resampling.LANCZOS)
        psd.create_pixel_layer(im, name=p["name"], top=int(p["y"]), left=int(p["x"]))
    psd.save(str(PSD_PATH))


def _com_float(value) -> float:
    if hasattr(value, "Value"):
        return float(value.Value)
    return float(value)


def write_psd_photoshop_com(placed: list[dict], canvas_size: tuple[int, int]) -> bool:
    try:
        import win32com.client
    except ImportError:
        print("win32com not available")
        return False

    w, h = canvas_size
    ps_display_no_dialogs = 3
    ps_pixels = 1
    ps_new_rgb = 2
    ps_transparent = 3
    ps_do_not_save = 2

    app = None
    last_exc = None
    for attempt in range(6):
        try:
            app = win32com.client.Dispatch("Photoshop.Application")
            app.Visible = True
            app.DisplayDialogs = ps_display_no_dialogs
            app.Preferences.RulerUnits = ps_pixels
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            import time
            time.sleep(1.5)
    if app is None:
        print(f"Photoshop COM dispatch failed: {last_exc}")
        return False

    try:
        doc = app.Documents.Add(w, h, 72, "western_scene", ps_new_rgb, ps_transparent)
        dw = _com_float(doc.Width)
        dh = _com_float(doc.Height)
        if abs(dw - w) > 1 or abs(dh - h) > 1:
            raise RuntimeError(f"Photoshop doc size {dw}x{dh} != {w}x{h}")

        def place_png(path: Path, name: str, x: int, y: int, scale: float) -> None:
            src = app.Open(str(path))
            src.ActiveLayer.Duplicate(doc)
            src.Close(ps_do_not_save)
            layer = doc.ActiveLayer
            layer.Name = name
            if abs(scale - 1.0) > 0.001:
                layer.Resize(scale * 100.0, scale * 100.0, 7)
            bounds = layer.Bounds
            cur_x = _com_float(bounds[0])
            cur_y = _com_float(bounds[1])
            layer.Translate(x - cur_x, y - cur_y)

        try:
            doc.ArtLayers["Layer 1"].Delete()
        except Exception:
            pass
        place_png(BG_PATH, "background", 0, 0, 1.0)
        for p in placed:
            png = PARTS_DIR / p["filename"]
            place_png(png, p["name"], int(p["x"]), int(p["y"]), float(p["scale"]))

        options = win32com.client.Dispatch("Photoshop.PhotoshopSaveOptions")
        options.EmbedColorProfile = True
        options.AlphaChannels = True
        options.Layers = True
        doc.SaveAs(str(PSD_PATH), options, True)
        nlayers = int(doc.ArtLayers.Count)
        print(f"Photoshop COM layers={nlayers} size={int(dw)}x{int(dh)}")
        return nlayers >= 2
    except Exception as exc:
        print(f"Photoshop COM place/save failed: {exc}")
        try:
            if app.Documents.Count:
                app.ActiveDocument.Close(ps_do_not_save)
        except Exception:
            pass
        return False


def verify_psd(path: Path) -> dict:
    data = path.read_bytes()[:4]
    header = data.decode("ascii", errors="replace")
    info = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "header": header,
        "valid_8bps": header == "8BPS",
    }
    if info["valid_8bps"]:
        psd = PSDImage.open(str(path))
        removed = False
        for layer in list(psd):
            if layer.name == "Layer 1":
                psd.remove(layer)
                removed = True
        if removed:
            psd.save(str(path))
            psd = PSDImage.open(str(path))
            info["bytes"] = path.stat().st_size
        info["size"] = list(psd.size)
        info["layer_count"] = len(list(psd))
        info["layer_names"] = [layer.name for layer in psd]
    return info


def main() -> int:
    from_placement = "--from-placement" in sys.argv
    bg = Image.open(BG_PATH)
    ref = Image.open(REF_PATH)
    canvas_size = bg.size
    print(f"bg={bg.size} {bg.mode}  ref={ref.size} {ref.mode}")
    if ref.size != bg.size:
        print("WARNING: reference size != background size; matching on reference pixels")
        canvas_size = ref.size

    if from_placement and PLACEMENT_PATH.is_file():
        payload = json.loads(PLACEMENT_PATH.read_text(encoding="utf-8"))
        placed = payload["parts"]
        placed.sort(key=layer_order)
        payload["parts"] = placed
        PLACEMENT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"loaded {len(placed)} placements from {PLACEMENT_PATH}")
    else:
        man_path = PARTS_DIR / "manifest.json"
        if not man_path.is_file():
            raise SystemExit(f"missing {man_path}; run split_sheet.py first")
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        parts = manifest["parts"]
        print(f"parts={len(parts)} from {man_path}")
        placed = match_all(parts)
        payload = {
            "background": str(BG_PATH),
            "reference": str(REF_PATH),
            "canvas": {"w": canvas_size[0], "h": canvas_size[1]},
            "part_count": len(placed),
            "parts": placed,
        }
        PLACEMENT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {PLACEMENT_PATH}")

    preview = composite_preview(placed, canvas_size)
    preview.convert("RGB").save(PREVIEW_PATH)
    print(f"wrote {PREVIEW_PATH} {preview.size}")

    com_ok = write_psd_photoshop_com(placed, canvas_size)
    if not com_ok:
        print("COM failed or incomplete; writing layered PSD via psd-tools")
        write_psd_tools(placed, canvas_size)
    elif not PSD_PATH.is_file() or PSD_PATH.stat().st_size < 100:
        print("COM reported success but PSD missing; writing via psd-tools")
        write_psd_tools(placed, canvas_size)

    info = verify_psd(PSD_PATH)
    print("psd", json.dumps(info, indent=2))

    xs = [p["x"] for p in placed]
    ys = [p["y"] for p in placed]
    print(f"placement x range {min(xs)}..{max(xs)}  y range {min(ys)}..{max(ys)}")
    origin = sum(1 for p in placed if p["x"] == 0 and p["y"] == 0)
    print(f"stacked_at_origin={origin}")
    weak = [p for p in placed if p["score"] < WEAK_SCORE]
    print(f"weak_matches={len(weak)}")
    for p in placed:
        print(
            f"  {p['name']:16s}  {p['filename']:14s}  "
            f"({p['x']:4d},{p['y']:4d})  {p['w']}x{p['h']}  "
            f"s={p['scale']:.2f}  score={p['score']:.3f}  "
            f"c={p.get('ccoeff', 0):.3f}  r={p.get('resid_ov', 0):.2f}"
        )
    return 0 if info.get("valid_8bps") else 1


if __name__ == "__main__":
    raise SystemExit(main())
