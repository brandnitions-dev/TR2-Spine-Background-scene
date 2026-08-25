#target photoshop
var destDir = new Folder("C:/Users/Emex33/Documents/fire frame vfx/backgroundSPINE/psd2-parts");
if (!destDir.exists) destDir.create();

function findDoc() {
  for (var i = 0; i < app.documents.length; i++) {
    if (String(app.documents[i].name).toLowerCase().indexOf("western_scene2") >= 0) {
      return app.documents[i];
    }
  }
  return app.activeDocument;
}

function px(u) {
  try { return u.as("px"); } catch (e) { return Number(u); }
}

function kindName(layer) {
  try {
    if (layer.kind === LayerKind.SMARTOBJECT) return "smart_object";
    if (layer.kind === LayerKind.TEXT) return "text";
    if (layer.kind === LayerKind.NORMAL) return "raster";
    return String(layer.kind);
  } catch (e) {
    return "unknown";
  }
}

function soSize(doc, layer) {
  var out = { w: 0, h: 0, ok: false, err: "" };
  if (layer.kind !== LayerKind.SMARTOBJECT) return out;
  var before = app.documents.length;
  try {
    app.activeDocument = doc;
    doc.activeLayer = layer;
    executeAction(stringIDToTypeID("placedLayerEditContents"), undefined, DialogModes.NO);
    if (app.documents.length > before) {
      var so = app.activeDocument;
      out.w = px(so.width);
      out.h = px(so.height);
      out.ok = true;
      so.close(SaveOptions.DONOTSAVECHANGES);
      app.activeDocument = doc;
    }
  } catch (e) {
    out.err = String(e);
    try { if (app.documents.length > before) app.activeDocument.close(SaveOptions.DONOTSAVECHANGES); } catch (e2) {}
    try { app.activeDocument = doc; } catch (e3) {}
  }
  return out;
}

var src = findDoc();
app.activeDocument = src;
app.displayDialogs = DialogModes.NO;
app.preferences.rulerUnits = Units.PIXELS;

var lines = [];
lines.push("doc=" + src.name);
lines.push("canvas=" + px(src.width) + "x" + px(src.height));

function walk(container, parent, depth) {
  for (var i = 0; i < container.layers.length; i++) {
    var ly = container.layers[i];
    if (ly.typename === "LayerSet") {
      lines.push("GROUP|" + ly.name + "|parent=" + parent + "|vis=" + ly.visible);
      walk(ly, ly.name, depth + 1);
    } else {
      var b = ly.bounds;
      var kind = kindName(ly);
      var so = soSize(src, ly);
      lines.push(
        "LAYER|" + ly.name +
        "|kind=" + kind +
        "|vis=" + ly.visible +
        "|op=" + ly.opacity +
        "|blend=" + ly.blendMode +
        "|bounds=" + px(b[0]) + "," + px(b[1]) + "," + px(b[2]) + "," + px(b[3]) +
        "|so=" + so.w + "x" + so.h +
        "|so_ok=" + so.ok +
        "|parent=" + parent
      );
    }
  }
}

walk(src, "", 0);

var rf = new File(destDir.fsName + "/hq_dump.txt");
rf.encoding = "UTF-8";
rf.open("w");
rf.write(lines.join("\n"));
rf.close();
