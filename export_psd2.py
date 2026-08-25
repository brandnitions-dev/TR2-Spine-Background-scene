"""
Dump western_scene2.psd layers from the live Photoshop document and export
trimmed RGBA PNGs. Photoshop bounds are ground truth (no template-match).

  python backgroundSPINE/export_psd2.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "psd2_layers.json"
PARTS_DIR = HERE / "psd2-parts"
PSD_COPY = HERE / "western_scene2.psd"
DESKTOP_PSD = Path(r"C:\Users\Emex33\Desktop\western_scene2.psd")

PS_DISPLAY_NO_DIALOGS = 3
PS_PIXELS = 1
PS_DO_NOT_SAVE = 2

BLEND_NAMES = {
    2: "normal",
    3: "dissolve",
    4: "darken",
    5: "multiply",
    6: "colorBurn",
    7: "linearBurn",
    8: "darkerColor",
    9: "lighten",
    10: "screen",
    11: "colorDodge",
    12: "linearDodge",
    13: "lighterColor",
    14: "overlay",
    15: "softLight",
    16: "hardLight",
    17: "vividLight",
    18: "linearLight",
    19: "pinLight",
    20: "hardMix",
    21: "difference",
    22: "exclusion",
    23: "blendSubtraction",
    24: "blendDivide",
    25: "hue",
    26: "saturation",
    27: "color",
    28: "luminosity",
}


def com_float(value) -> float:
    if hasattr(value, "Value"):
        return float(value.Value)
    return float(value)


def layer_typename(layer) -> str:
    for attr in ("Typename", "typename"):
        try:
            raw = getattr(layer, attr)
            if raw:
                return str(raw)
        except Exception:
            continue
    try:
        _ = layer.Layers
        return "LayerSet"
    except Exception:
        return "ArtLayer"


def layer_bounds(layer) -> list[float] | None:
    try:
        b = layer.Bounds
        return [round(com_float(b[i]), 3) for i in range(4)]
    except Exception:
        return None


def walk_layers(container, parent: str, depth: int, order: list[int]) -> list[dict]:
    layers = container.Layers
    count = int(layers.Count)
    rows: list[dict] = []
    for i in range(1, count + 1):
        layer = layers.Item(i)
        order[0] += 1
        name = str(layer.Name)
        typename = layer_typename(layer)
        is_set = typename == "LayerSet"
        blend_raw = None
        blend_name = None
        opacity = None
        visible = True
        try:
            visible = bool(layer.Visible)
        except Exception:
            pass
        try:
            opacity = round(com_float(layer.Opacity), 3)
        except Exception:
            pass
        try:
            blend_raw = int(layer.BlendMode)
            blend_name = BLEND_NAMES.get(blend_raw, str(blend_raw))
        except Exception:
            pass
        bounds = None if is_set else layer_bounds(layer)
        ps_kind = None
        if not is_set:
            try:
                raw_kind = int(layer.Kind)
            except Exception:
                raw_kind = None
            ps_kind = {
                1: "raster",
                2: "text",
                17: "smart_object",
            }.get(raw_kind, str(raw_kind) if raw_kind is not None else "unknown")
        rec = {
            "index": order[0],
            "stack_index": i,
            "name": name,
            "safe_name": sanitize(name, order[0]),
            "parent": parent,
            "depth": depth,
            "kind": "group" if is_set else "art",
            "ps_kind": ps_kind,
            "typename": typename,
            "visible": visible,
            "opacity": opacity,
            "blend": blend_name,
            "blend_raw": blend_raw,
            "bounds": bounds,
        }
        if bounds and len(bounds) == 4:
            left, top, right, bottom = bounds
            rec["width"] = round(right - left, 3)
            rec["height"] = round(bottom - top, 3)
        rows.append(rec)
        if is_set:
            rows.extend(walk_layers(layer, name, depth + 1, order))
    return rows


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


def connect_app():
    import win32com.client

    last = None
    for _ in range(8):
        try:
            app = win32com.client.Dispatch("Photoshop.Application")
            app.Visible = True
            app.DisplayDialogs = PS_DISPLAY_NO_DIALOGS
            app.Preferences.RulerUnits = PS_PIXELS
            return app
        except Exception as exc:
            last = exc
            time.sleep(1.2)
    raise SystemExit(f"Photoshop COM dispatch failed: {last}")


def find_scene2(app):
    docs = []
    count = int(app.Documents.Count)
    for i in range(1, count + 1):
        doc = app.Documents.Item(i)
        rec = {
            "name": str(doc.Name),
            "full": None,
            "w": round(com_float(doc.Width), 3),
            "h": round(com_float(doc.Height), 3),
        }
        try:
            rec["full"] = str(doc.FullName)
        except Exception:
            rec["full"] = None
        docs.append((doc, rec))
    match = None
    for doc, rec in docs:
        name = rec["name"].lower()
        full = (rec["full"] or "").lower()
        if "western_scene2" in name or "western_scene2" in full:
            match = (doc, rec)
            break
    return docs, match


def save_copy(doc, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    import win32com.client

    options = win32com.client.Dispatch("Photoshop.PhotoshopSaveOptions")
    options.EmbedColorProfile = True
    options.AlphaChannels = True
    options.Layers = True
    doc.SaveAs(str(dest), options, True)


def main() -> int:
    app = connect_app()
    docs, match = find_scene2(app)
    print("open documents:")
    for _, rec in docs:
        print(f"  {rec['name']}  {rec['w']}x{rec['h']}  {rec['full']}")

    if match is None:
        if DESKTOP_PSD.is_file():
            print(f"opening {DESKTOP_PSD}")
            doc = app.Open(str(DESKTOP_PSD))
            rec = {
                "name": str(doc.Name),
                "full": str(DESKTOP_PSD),
                "w": round(com_float(doc.Width), 3),
                "h": round(com_float(doc.Height), 3),
            }
            match = (doc, rec)
        else:
            raise SystemExit("western_scene2.psd is not open and not on Desktop")

    doc, rec = match
    app.ActiveDocument = doc
    canvas_w = int(round(com_float(doc.Width)))
    canvas_h = int(round(com_float(doc.Height)))
    print(f"active {rec['name']} canvas={canvas_w}x{canvas_h}")

    layers = walk_layers(doc, "", 0, [0])
    payload = {
        "source_open": rec["name"],
        "source_full": rec["full"],
        "saved_copy": str(PSD_COPY),
        "canvas": {"w": canvas_w, "h": canvas_h},
        "mode": str(getattr(doc, "Mode", "")),
        "layer_count": len(layers),
        "layers": layers,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON} layers={len(layers)}")
    for layer in layers:
        b = layer.get("bounds")
        print(
            f"  [{layer['index']:02d}] {layer['kind']:5s} vis={int(layer['visible'])} "
            f"op={layer['opacity']} blend={layer['blend']} "
            f"{layer['name']!r} bounds={b} parent={layer['parent']!r}"
        )

    try:
        save_copy(doc, PSD_COPY)
        print(f"saved copy {PSD_COPY} bytes={PSD_COPY.stat().st_size}")
        payload["saved_copy_bytes"] = PSD_COPY.stat().st_size
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"SaveAs copy failed ({exc}); file-copy fallback")
        if DESKTOP_PSD.is_file():
            import shutil

            shutil.copy2(DESKTOP_PSD, PSD_COPY)
            print(f"copied {DESKTOP_PSD} -> {PSD_COPY}")
        else:
            print("WARNING: could not save PSD copy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
