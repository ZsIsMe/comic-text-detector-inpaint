/*
Add configurable image layers to existing PSD files.

Run in Photoshop:
File > Scripts > Browse... > add_layers_to_existing_psds.jsx

For every PSD in the selected folder, each configured source folder is searched
for a file with the same name stem. The matched file is imported as a layer,
renamed, assigned the configured visibility, and the original PSD is overwritten.
Source and PSD dimensions do not need to match.
*/

#target photoshop

(function () {
    app.bringToFront();

    var oldRulerUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    try {
        var settings = showSettingsDialog();
        if (!settings) return;

        var psdFolder = new Folder(settings.psdFolder);
        if (!psdFolder.exists) {
            alert("PSD 文件夹不存在：\n" + psdFolder.fsName);
            return;
        }

        var psdFiles = psdFolder.getFiles(function (file) {
            return file instanceof File && /\.psd$/i.test(file.name);
        });
        psdFiles.sort(function (a, b) {
            return naturalCompareNames(a.name, b.name);
        });

        if (psdFiles.length === 0) {
            alert("PSD 文件夹中没有找到 PSD 文件：\n" + psdFolder.fsName);
            return;
        }

        var added = 0;
        var processed = 0;
        var skipped = [];
        var failed = [];

        for (var i = 0; i < psdFiles.length; i++) {
            var psdFile = psdFiles[i];
            var doc = null;
            var addedForPsd = 0;
            var skippedForPsd = [];

            try {
                doc = app.open(psdFile);

                for (var configIndex = 0; configIndex < settings.extraLayers.length; configIndex++) {
                    var config = settings.extraLayers[configIndex];
                    var sourceFile = findSourceFile(config.folder, stripExtension(psdFile.name));
                    if (!sourceFile) {
                        skippedForPsd.push(config.layerName + "：找不到匹配图片");
                        continue;
                    }

                    try {
                        importLayer(doc, sourceFile, config.layerName, config.visible);
                        added++;
                        addedForPsd++;
                    } catch (layerErr) {
                        skippedForPsd.push(config.layerName + "：" + layerErr.message);
                    }
                }

                app.activeDocument = doc;
                doc.save();
                processed++;

                if (skippedForPsd.length > 0) {
                    skipped.push(psdFile.name + "：" + skippedForPsd.join("；"));
                }
            } catch (err) {
                failed.push(psdFile.name + "：" + err.message);
            } finally {
                if (doc) {
                    try {
                        app.activeDocument = doc;
                        doc.close(SaveOptions.DONOTSAVECHANGES);
                    } catch (closeErr) {
                    }
                }
            }
        }

        var reportFile = new File(psdFolder.fsName + "/add_layers_report.txt");
        writeReport(
            reportFile,
            psdFolder,
            psdFiles.length,
            processed,
            added,
            settings,
            skipped,
            failed
        );

        var message = "PSD 图层添加完成：\n" +
            "处理 PSD：" + processed + " / " + psdFiles.length + "\n" +
            "添加图层：" + added + " 个\n" +
            "已覆盖原 PSD：是";
        if (skipped.length > 0) {
            message += "\n跳过图层：" + skipped.length + " 个 PSD";
        }
        if (failed.length > 0) {
            message += "\n处理失败：" + failed.length + " 个 PSD";
        }
        message += "\n报告：\n" + reportFile.fsName;
        alert(message);
    } catch (e) {
        alert("Add layers to existing PSDs failed:\n" + e.toString() + "\nLine: " + (e.line || "unknown"));
    } finally {
        app.preferences.rulerUnits = oldRulerUnits;
    }

    function showSettingsDialog() {
        var dialog = new Window("dialog", "为已有 PSD 添加图层");
        dialog.orientation = "column";
        dialog.alignChildren = ["fill", "top"];
        dialog.spacing = 10;
        dialog.margins = 16;

        var psdGroup = dialog.add("group");
        psdGroup.orientation = "row";
        psdGroup.alignChildren = ["fill", "center"];
        psdGroup.add("statictext", undefined, "PSD 文件夹：");
        var psdPathInput = psdGroup.add("edittext", undefined, "");
        psdPathInput.characters = 52;
        var psdBrowseButton = psdGroup.add("button", undefined, "选择");

        var extraLayersPanel = dialog.add("panel", undefined, "额外图层");
        extraLayersPanel.orientation = "column";
        extraLayersPanel.alignChildren = ["fill", "top"];
        extraLayersPanel.spacing = 6;

        var extraLayersRows = extraLayersPanel.add("group");
        extraLayersRows.orientation = "column";
        extraLayersRows.alignChildren = ["fill", "top"];
        extraLayersRows.spacing = 4;

        var extraLayerConfigs = [];
        var addExtraLayerButton = extraLayersPanel.add("button", undefined, "添加文件夹");

        function addExtraLayerRow(folderPath, layerName, visible) {
            var row = extraLayersRows.add("group");
            row.orientation = "row";
            row.alignChildren = ["fill", "center"];

            var folderInput = row.add("edittext", undefined, folderPath || "");
            folderInput.characters = 34;
            var browseButton = row.add("button", undefined, "选择文件夹");
            var layerInput = row.add("edittext", undefined, layerName || "");
            layerInput.characters = 16;
            var visibleCheckbox = row.add("checkbox", undefined, "显示");
            visibleCheckbox.value = visible !== false;
            var removeButton = row.add("button", undefined, "删除");

            browseButton.onClick = function () {
                var selected = Folder.selectDialog("选择额外图层图片文件夹");
                if (selected) folderInput.text = selected.fsName;
            };

            removeButton.onClick = function () {
                row.parent.remove(row);
                for (var configIndex = extraLayerConfigs.length - 1; configIndex >= 0; configIndex--) {
                    if (extraLayerConfigs[configIndex].row === row) {
                        extraLayerConfigs.splice(configIndex, 1);
                        break;
                    }
                }
                refreshExtraLayersLayout();
            };

            extraLayerConfigs.push({
                row: row,
                folderInput: folderInput,
                layerInput: layerInput,
                visibleCheckbox: visibleCheckbox
            });
            refreshExtraLayersLayout();
        }

        addExtraLayerButton.onClick = function () {
            addExtraLayerRow("", "", true);
        };

        function refreshExtraLayersLayout() {
            extraLayersRows.preferredSize = [-1, -1];
            extraLayersPanel.preferredSize = [-1, -1];
            dialog.preferredSize = [-1, -1];
            extraLayersRows.layout.layout(true);
            extraLayersPanel.layout.layout(true);
            dialog.layout.layout(true);
            dialog.size = dialog.preferredSize;
            dialog.layout.resize();
        }

        var buttonGroup = dialog.add("group");
        buttonGroup.alignment = "right";
        buttonGroup.add("button", undefined, "Cancel", { name: "cancel" });
        var okButton = buttonGroup.add("button", undefined, "OK", { name: "ok" });

        psdBrowseButton.onClick = function () {
            var selected = Folder.selectDialog("选择已有 PSD 文件夹");
            if (selected) psdPathInput.text = selected.fsName;
        };

        okButton.onClick = function () {
            if (!trimString(psdPathInput.text)) {
                alert("请选择 PSD 文件夹。");
                return;
            }

            var selectedPsdFolder = new Folder(trimString(psdPathInput.text));
            if (!selectedPsdFolder.exists) {
                alert("PSD 文件夹不存在：\n" + selectedPsdFolder.fsName);
                return;
            }

            var extraLayers = [];
            for (var configIndex = 0; configIndex < extraLayerConfigs.length; configIndex++) {
                var extraConfig = extraLayerConfigs[configIndex];
                var folderPath = trimString(extraConfig.folderInput.text);
                var layerName = trimString(extraConfig.layerInput.text);
                if (!folderPath && !layerName) continue;
                if (!folderPath || !layerName) {
                    alert("额外图层配置必须同时填写文件夹和图层名。");
                    return;
                }

                var extraFolder = new Folder(folderPath);
                if (!extraFolder.exists) {
                    alert("额外图层文件夹不存在：\n" + folderPath);
                    return;
                }

                extraLayers.push({
                    folder: extraFolder,
                    layerName: layerName,
                    visible: extraConfig.visibleCheckbox.value
                });
            }

            if (extraLayers.length === 0) {
                alert("请至少添加一个额外图层文件夹。");
                return;
            }

            dialog.extraLayers = extraLayers;
            dialog.close(1);
        };

        if (dialog.show() !== 1) return null;

        return {
            psdFolder: trimString(psdPathInput.text),
            extraLayers: dialog.extraLayers || []
        };
    }

    function importLayer(targetDoc, sourceFile, layerName, visible) {
        var sourceDoc = null;
        try {
            sourceDoc = app.open(sourceFile);

            app.activeDocument = sourceDoc;
            if (sourceDoc.mode !== DocumentMode.RGB) {
                sourceDoc.changeMode(ChangeMode.RGB);
            }
            if (sourceDoc.layers.length > 1) {
                sourceDoc.mergeVisibleLayers();
            }

            var sourceLayer = sourceDoc.activeLayer;
            var sourceBounds = getLayerBounds(sourceLayer);
            var importedLayer = sourceLayer.duplicate(targetDoc, ElementPlacement.PLACEATBEGINNING);

            app.activeDocument = targetDoc;
            targetDoc.activeLayer = importedLayer;
            importedLayer.name = layerName;
            alignLayerBounds(importedLayer, sourceBounds);
            importedLayer.visible = visible;

            moveLayerAboveBackground(targetDoc, importedLayer);
        } finally {
            if (sourceDoc) {
                try {
                    app.activeDocument = sourceDoc;
                    sourceDoc.close(SaveOptions.DONOTSAVECHANGES);
                } catch (closeErr) {
                }
            }
        }
    }

    function moveLayerAboveBackground(doc, layer) {
        try {
            var bgLayer = doc.artLayers.getByName("bg");
            layer.move(bgLayer, ElementPlacement.PLACEBEFORE);
        } catch (e) {
        }
    }

    function getLayerBounds(layer) {
        return {
            left: Math.round(layer.bounds[0].as("px")),
            top: Math.round(layer.bounds[1].as("px")),
            right: Math.round(layer.bounds[2].as("px")),
            bottom: Math.round(layer.bounds[3].as("px"))
        };
    }

    function alignLayerBounds(layer, expectedBounds) {
        var actualBounds = getLayerBounds(layer);
        var dx = expectedBounds.left - actualBounds.left;
        var dy = expectedBounds.top - actualBounds.top;
        if (dx !== 0 || dy !== 0) layer.translate(dx, dy);
    }

    function findSourceFile(folder, stem) {
        var extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".psd"];
        for (var i = 0; i < extensions.length; i++) {
            var file = new File(folder.fsName + "/" + stem + extensions[i]);
            if (file.exists) return file;
        }
        return null;
    }

    function writeReport(reportFile, psdFolder, total, processed, added, settings, skipped, failed) {
        reportFile.encoding = "UTF-8";
        if (!reportFile.open("w")) {
            alert("无法写入报告：\n" + reportFile.fsName);
            return;
        }

        reportFile.writeln("Add layers to existing PSDs report");
        reportFile.writeln("Generated at: " + formatDate(new Date()));
        reportFile.writeln("PSD folder: " + psdFolder.fsName);
        reportFile.writeln("Total PSD files: " + total);
        reportFile.writeln("Processed PSD files: " + processed);
        reportFile.writeln("Added layers: " + added);
        reportFile.writeln("Overwrite original PSD: yes");
        reportFile.writeln("");
        reportFile.writeln("[CONFIGURED_LAYERS]");
        for (var i = 0; i < settings.extraLayers.length; i++) {
            var config = settings.extraLayers[i];
            reportFile.writeln(
                config.layerName + " | visible=" + (config.visible ? "yes" : "no") +
                " | folder=" + config.folder.fsName
            );
        }
        reportFile.writeln("");
        reportFile.writeln("[SKIPPED_LAYERS]");
        writeLines(reportFile, skipped);
        reportFile.writeln("");
        reportFile.writeln("[FAILED_PSDS]");
        writeLines(reportFile, failed);
        reportFile.close();
    }

    function writeLines(file, lines) {
        if (lines.length === 0) {
            file.writeln("(none)");
            return;
        }
        for (var i = 0; i < lines.length; i++) file.writeln(lines[i]);
    }

    function stripExtension(name) {
        return name.replace(/\.[^\.]+$/, "");
    }

    function naturalCompareNames(a, b) {
        var ax = splitNaturalName(a);
        var bx = splitNaturalName(b);
        var len = Math.min(ax.length, bx.length);
        for (var i = 0; i < len; i++) {
            if (ax[i][0] !== bx[i][0]) return ax[i][0] < bx[i][0] ? -1 : 1;
            if (ax[i][1] !== bx[i][1]) return ax[i][1] < bx[i][1] ? -1 : 1;
            if (ax[i].length > 2 && bx[i].length > 2 && ax[i][2] !== bx[i][2]) {
                return ax[i][2] < bx[i][2] ? -1 : 1;
            }
        }
        if (ax.length !== bx.length) return ax.length < bx.length ? -1 : 1;
        a = a.toLowerCase();
        b = b.toLowerCase();
        return a < b ? -1 : (a > b ? 1 : 0);
    }

    function splitNaturalName(name) {
        var parts = [];
        var lowerName = name.toLowerCase();
        var pattern = /\d+/g;
        var lastIndex = 0;
        var match;
        while ((match = pattern.exec(lowerName)) !== null) {
            parts.push(lowerName.substring(lastIndex, match.index));
            parts.push(match[0]);
            lastIndex = pattern.lastIndex;
        }
        parts.push(lowerName.substring(lastIndex));

        var result = [];
        for (var i = 0; i < parts.length; i++) {
            if (/^\d+$/.test(parts[i])) {
                result.push([1, parseInt(parts[i], 10), parts[i]]);
            } else {
                result.push([0, parts[i]]);
            }
        }
        return result;
    }

    function trimString(value) {
        return value.replace(/^\s+|\s+$/g, "");
    }

    function pad2(value) {
        return value < 10 ? "0" + value : String(value);
    }

    function formatDate(date) {
        return date.getFullYear() + "-" +
            pad2(date.getMonth() + 1) + "-" +
            pad2(date.getDate()) + " " +
            pad2(date.getHours()) + ":" +
            pad2(date.getMinutes()) + ":" +
            pad2(date.getSeconds());
    }
})();
