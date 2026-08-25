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

function safeName(name, index) {
  var s = String(name).toLowerCase().replace(/[^\w\-]+/g, "_").replace(/^_+|_+$/g, "");
  if (!s) s = "layer_" + index;
  while (s.indexOf("__") >= 0) s = s.replace("__", "_");
  return s;
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

function hideAll(container) {
  for (var i = 0; i < container.layers.length; i++) {
    var ly = container.layers[i];
    ly.visible = false;
    if (ly.typename === "LayerSet") hideAll(ly);
  }
}

function showAncestors(layer) {
  var p = layer;
  while (p && p.typename !== "Document") {
    p.visible = true;
    p = p.parent;
  }
}

function collectArt(container, rows) {
  for (var i = 0; i < container.layers.length; i++) {
    var ly = container.layers[i];
    if (ly.typename === "LayerSet") collectArt(ly, rows);
    else rows.push(ly);
  }
}

function rasterizeTree(container) {
  for (var i = 0; i < container.layers.length; i++) {
    var ly = container.layers[i];
    if (ly.typename === "LayerSet") {
      rasterizeTree(ly);
    } else {
      try { ly.rasterize(RasterizeType.LAYERSTYLE); } catch (e1) {}
      try { ly.rasterize(RasterizeType.ENTIRELAYER); } catch (e2) {}
    }
  }
}

function showSelfAndBelow(arts, target) {
  var found = false;
  for (var i = 0; i < arts.length; i++) {
    if (arts[i] === target) found = true;
    if (found) showAncestors(arts[i]);
  }
}

var src = findDoc();
app.activeDocument = src;
app.displayDialogs = DialogModes.NO;
app.preferences.rulerUnits = Units.PIXELS;

var srcW = px(src.width);
var srcH = px(src.height);
var report = [];
report.push("doc=" + src.name);
report.push("src_canvas=" + srcW + "x" + srcH);
report.push("export_canvas=" + srcW + "x" + srcH);
report.push("scale=1");
report.push("effects=rasterize_layerstyle+entirelayer");
report.push("smart_objects=none_reuse_live_raster");

var srcArts = [];
collectArt(src, srcArts);
for (var i = 0; i < srcArts.length; i++) {
  var layer = srcArts[i];
  var nm = safeName(layer.name, i);
  var kind = kindName(layer);
  var b0 = layer.bounds;
  report.push(
    "meta " + nm +
    " kind=" + kind +
    " vis=" + layer.visible +
    " op=" + layer.opacity +
    " blend=" + layer.blendMode +
    " canvas_bounds=" + px(b0[0]) + "," + px(b0[1]) + "," + px(b0[2]) + "," + px(b0[3]) +
    " name=" + layer.name
  );
}

var work = src.duplicate("psd2_hq_export", false);
app.activeDocument = work;
rasterizeTree(work);

var pngOpts = new PNGSaveOptions();
pngOpts.compression = 6;
pngOpts.interlaced = false;

var arts = [];
collectArt(work, arts);
var usedNames = {};

function uniqueSafe(name, index) {
  var base = safeName(name, index);
  if (!usedNames[base]) {
    usedNames[base] = 1;
    return base;
  }
  usedNames[base] += 1;
  var n = usedNames[base];
  var suffix = (n < 10 ? "0" : "") + n;
  return base + "_" + suffix;
}

for (var j = 0; j < arts.length; j++) {
  var art = arts[j];
  var b = art.bounds;
  var l = px(b[0]);
  var t = px(b[1]);
  var r = px(b[2]);
  var bot = px(b[3]);
  var w = r - l;
  var h = bot - t;
  var an = uniqueSafe(art.name, j);
  if (w < 1 || h < 1) {
    report.push("skip " + an + " empty");
    continue;
  }
  var blendName = String(art.blendMode);
  hideAll(work);
  showAncestors(art);
  var file = new File(destDir.fsName + "/full_" + an + ".png");
  work.saveAs(file, pngOpts, true, Extension.LOWERCASE);
  report.push("full " + an + " " + l + "," + t + "," + r + "," + bot + " name=" + art.name);

  if (blendName !== "BlendMode.NORMAL") {
    hideAll(work);
    showSelfAndBelow(arts, art);
    var stack = new File(destDir.fsName + "/stack_" + an + ".png");
    work.saveAs(stack, pngOpts, true, Extension.LOWERCASE);
    report.push("stack " + an + " blend=" + blendName);
  }
}

work.close(SaveOptions.DONOTSAVECHANGES);
app.activeDocument = src;

var rf = new File(destDir.fsName + "/export_log.txt");
rf.encoding = "UTF-8";
rf.open("w");
rf.write(report.join("\n"));
rf.close();
