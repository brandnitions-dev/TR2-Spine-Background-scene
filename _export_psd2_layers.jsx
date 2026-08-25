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

var src = findDoc();
app.activeDocument = src;
app.displayDialogs = DialogModes.NO;
app.preferences.rulerUnits = Units.PIXELS;

var work = src.duplicate("psd2_export_work", false);
app.activeDocument = work;

var pngOpts = new PNGSaveOptions();
pngOpts.compression = 6;
pngOpts.interlaced = false;

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
    if (ly.typename === "LayerSet") {
      collectArt(ly, rows);
    } else {
      rows.push(ly);
    }
  }
}

function safeName(name, index) {
  var s = String(name).toLowerCase().replace(/[^\w\-]+/g, "_").replace(/^_+|_+$/g, "");
  if (!s) s = "layer_" + index;
  return s;
}

var arts = [];
collectArt(work, arts);
var report = [];
report.push("doc=" + src.name);
report.push("work_layers=" + arts.length);

for (var i = 0; i < arts.length; i++) {
  var layer = arts[i];
  var b = layer.bounds;
  var l = b[0].as("px");
  var t = b[1].as("px");
  var r = b[2].as("px");
  var bot = b[3].as("px");
  var w = r - l;
  var h = bot - t;
  var nm = safeName(layer.name, i);
  if (w < 1 || h < 1) {
    report.push("skip " + nm + " empty");
    continue;
  }
  hideAll(work);
  showAncestors(layer);
  var file = new File(destDir.fsName + "/full_" + nm + ".png");
  work.saveAs(file, pngOpts, true, Extension.LOWERCASE);
  report.push("full " + nm + " " + l + "," + t + "," + r + "," + bot);
}

work.close(SaveOptions.DONOTSAVECHANGES);
app.activeDocument = src;

var rf = new File(destDir.fsName + "/export_log.txt");
rf.encoding = "UTF-8";
rf.open("w");
rf.write(report.join("\n"));
rf.close();
