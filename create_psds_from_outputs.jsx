/*
Create PSD files from solid_inpaint outputs.

Run in Photoshop:
File > Scripts > Browse... > create_psds_from_outputs.jsx

Expected input layout:
<image folder>/ctd_inpainted/other_mask/<name>.png
<image folder>/ctd_inpainted/inpainted/<name>.png

Output:
<image folder>/ctd_inpainted/psd/<name>.psd

Each PSD contains:
- bg
- overlay-manual
- OTHER_CHANNEL, or hidden layer OTHER_CHANNEL when enabled
*/

#target photoshop

(function () {
    app.bringToFront();

    var oldRulerUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;

    try {
        var settings = showSettingsDialog();
        if (!settings) return;

        var imageFolder = new Folder(settings.imageFolder);
        var outputRoot = new Folder(settings.outputRoot);
        var otherMaskFolder = new Folder(outputRoot.fsName + "/other_mask");
        var overlayFolder = new Folder(outputRoot.fsName + "/inpainted");
        var psdFolder = new Folder(outputRoot.fsName + "/psd");

        if (!imageFolder.exists) {
            alert("原图文件夹不存在：\n" + imageFolder.fsName);
            return;
        }
        if (!otherMaskFolder.exists) {
            alert("other_mask 文件夹不存在：\n" + otherMaskFolder.fsName);
            return;
        }
        if (!overlayFolder.exists) {
            alert("inpainted 文件夹不存在：\n" + overlayFolder.fsName);
            return;
        }
        if (!psdFolder.exists) {
            psdFolder.create();
        }

        var imageFiles = imageFolder.getFiles(function (file) {
            if (!(file instanceof File)) return false;
            return /\.(png|jpg|jpeg|tif|tiff|bmp|psd)$/i.test(file.name);
        });
        imageFiles.sort(function (a, b) {
            return naturalCompareNames(a.name, b.name);
        });

        var made = 0;
        var actionRun = [];
        var actionErrors = [];
        var skippedExisting = [];
        var skipped = [];

        for (var i = 0; i < imageFiles.length; i++) {
            var imageFile = imageFiles[i];
            var stem = stripExtension(imageFile.name);
            var psdFile = new File(psdFolder.fsName + "/" + stem + ".psd");
            if (!settings.restart && psdFile.exists) {
                skippedExisting.push(imageFile.name + "：目标 PSD 已存在");
                continue;
            }

            var otherMaskFile = findMaskFile(otherMaskFolder, stem);
            var overlayFile = findMaskFile(overlayFolder, stem);

            if (!otherMaskFile) {
                skipped.push(imageFile.name + "：缺少 other_mask");
                continue;
            }
            if (!overlayFile) {
                skipped.push(imageFile.name + "：缺少 inpainted overlay");
                continue;
            }

            var doc = null;
            try {
                doc = createDocument(imageFile);
                importOverlayLayer(doc, overlayFile, "overlay-manual");
                importMaskAsAlpha(doc, otherMaskFile, "OTHER_CHANNEL", true);

                if (settings.runAction && hasChannel(doc, "OTHER_CHANNEL")) {
                    try {
                        app.activeDocument = doc;
                        app.doAction(settings.actionName, settings.actionSetName);
                        actionRun.push(imageFile.name);
                    } catch (actionErr) {
                        actionErrors.push(imageFile.name + "：" + actionErr.message);
                    }
                }

                app.activeDocument = doc;
                if (settings.convertOtherChannelToLayer) {
                    convertAlphaChannelToHiddenLayer(doc, "OTHER_CHANNEL", "OTHER_CHANNEL");
                }
                setRGBChannels(doc);
                var saveOptions = new PhotoshopSaveOptions();
                saveOptions.alphaChannels = !settings.convertOtherChannelToLayer;
                saveOptions.layers = true;
                saveOptions.maximizeCompatibility = true;
                doc.saveAs(psdFile, saveOptions, true, Extension.LOWERCASE);
                made++;
            } catch (err) {
                skipped.push(imageFile.name + "：" + err.message);
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

        writeReport(
            new File(psdFolder.fsName + "/create_psds_report.txt"),
            imageFolder,
            outputRoot,
            imageFiles.length,
            made,
            settings,
            actionRun,
            actionErrors,
            skippedExisting,
            skipped
        );

        var message = "PSD 生成完成：" + made + " 个\n输出目录：\n" + psdFolder.fsName;
        if (settings.runAction) {
            message += "\n执行动作：" + actionRun.length + "\n动作失败：" + actionErrors.length;
        }
        if (skippedExisting.length > 0) {
            message += "\n已存在跳过：" + skippedExisting.length;
        }
        if (skipped.length > 0) {
            message += "\n\n跳过/失败：" + skipped.length + " 个\n" + skipped.slice(0, 20).join("\n");
            if (skipped.length > 20) message += "\n...";
        }
        if (actionErrors.length > 0) {
            message += "\n\n动作失败前 20 个：\n" + actionErrors.slice(0, 20).join("\n");
            if (actionErrors.length > 20) message += "\n...";
        }
        alert(message);
    } catch (e) {
        alert("Create solid inpaint PSDs failed:\n" + e.toString() + "\nLine: " + (e.line || "unknown"));
    } finally {
        app.preferences.rulerUnits = oldRulerUnits;
    }

    function showSettingsDialog() {
        var actionSets = getActionSets();
        var dialog = new Window("dialog", "生成 solid_inpaint PSD");
        dialog.orientation = "column";
        dialog.alignChildren = ["fill", "top"];
        dialog.spacing = 10;
        dialog.margins = 16;

        var imageGroup = dialog.add("group");
        imageGroup.orientation = "row";
        imageGroup.alignChildren = ["fill", "center"];
        imageGroup.add("statictext", undefined, "原图文件夹：");
        var imagePathInput = imageGroup.add("edittext", undefined, "");
        imagePathInput.characters = 52;
        var imageBrowseButton = imageGroup.add("button", undefined, "选择");

        var outputGroup = dialog.add("group");
        outputGroup.orientation = "row";
        outputGroup.alignChildren = ["fill", "center"];
        outputGroup.add("statictext", undefined, "ctd_inpainted：");
        var outputPathInput = outputGroup.add("edittext", undefined, "");
        outputPathInput.characters = 52;
        var outputBrowseButton = outputGroup.add("button", undefined, "选择");

        var restartGroup = dialog.add("group");
        restartGroup.orientation = "row";
        restartGroup.alignChildren = ["left", "center"];
        var restartCheckbox = restartGroup.add("checkbox", undefined, "重新开始（覆盖已有 PSD）");
        restartCheckbox.value = false;

        var actionEnableGroup = dialog.add("group");
        actionEnableGroup.orientation = "row";
        actionEnableGroup.alignChildren = ["left", "center"];
        var actionCheckbox = actionEnableGroup.add("checkbox", undefined, "有 OTHER_CHANNEL 时执行动作");
        actionCheckbox.value = false;

        var convertChannelGroup = dialog.add("group");
        convertChannelGroup.orientation = "row";
        convertChannelGroup.alignChildren = ["left", "center"];
        var convertOtherChannelCheckbox = convertChannelGroup.add("checkbox", undefined, "OTHER_CHANNEL 通道改为图层（动作后）");
        convertOtherChannelCheckbox.value = false;

        var actionSetGroup = dialog.add("group");
        actionSetGroup.orientation = "row";
        actionSetGroup.alignChildren = ["left", "center"];
        actionSetGroup.add("statictext", undefined, "动作组：");
        var setDropdown = actionSetGroup.add("dropdownlist", undefined, []);
        setDropdown.minimumSize.width = 260;

        var actionGroup = dialog.add("group");
        actionGroup.orientation = "row";
        actionGroup.alignChildren = ["left", "center"];
        actionGroup.add("statictext", undefined, "动作：");
        var actionDropdown = actionGroup.add("dropdownlist", undefined, []);
        actionDropdown.minimumSize.width = 260;

        var buttonGroup = dialog.add("group");
        buttonGroup.orientation = "row";
        buttonGroup.alignment = "right";
        var okButton = buttonGroup.add("button", undefined, "OK", { name: "ok" });
        buttonGroup.add("button", undefined, "Cancel", { name: "cancel" });

        for (var i = 0; i < actionSets.length; i++) {
            setDropdown.add("item", actionSets[i].name);
        }
        if (actionSets.length > 0) {
            setDropdown.selection = 0;
        }
        refreshActionDropdown();
        refreshActionControls();

        imageBrowseButton.onClick = function () {
            var selected = Folder.selectDialog("选择原图文件夹");
            if (selected) {
                imagePathInput.text = selected.fsName;
                outputPathInput.text = selected.fsName + "/ctd_inpainted";
            }
        };

        outputBrowseButton.onClick = function () {
            var selected = Folder.selectDialog("选择 ctd_inpainted 文件夹");
            if (selected) {
                outputPathInput.text = selected.fsName;
            }
        };

        actionCheckbox.onClick = function () {
            refreshActionControls();
        };

        setDropdown.onChange = function () {
            refreshActionDropdown();
        };

        okButton.onClick = function () {
            if (!trimString(imagePathInput.text)) {
                alert("请选择原图文件夹。");
                return;
            }
            if (!trimString(outputPathInput.text)) {
                alert("请选择 ctd_inpainted 文件夹。");
                return;
            }
            if (actionCheckbox.value && (!setDropdown.selection || !actionDropdown.selection)) {
                alert("请先在 Photoshop Actions 面板载入动作，并选择动作组和动作。");
                return;
            }
            dialog.close(1);
        };

        if (dialog.show() !== 1) return null;

        var selectedSet = setDropdown.selection ? actionSets[setDropdown.selection.index] : null;
        var selectedAction = selectedSet && actionDropdown.selection ? selectedSet.actions[actionDropdown.selection.index] : null;

        return {
            imageFolder: trimString(imagePathInput.text),
            outputRoot: trimString(outputPathInput.text),
            restart: restartCheckbox.value,
            runAction: actionCheckbox.value,
            convertOtherChannelToLayer: convertOtherChannelCheckbox.value,
            actionSetName: selectedSet ? selectedSet.name : "",
            actionName: selectedAction ? selectedAction.name : ""
        };

        function refreshActionDropdown() {
            actionDropdown.removeAll();
            if (!setDropdown.selection) return;
            var selectedSet = actionSets[setDropdown.selection.index];
            for (var j = 0; j < selectedSet.actions.length; j++) {
                actionDropdown.add("item", selectedSet.actions[j].name);
            }
            if (selectedSet.actions.length > 0) {
                actionDropdown.selection = 0;
            }
        }

        function refreshActionControls() {
            var enabled = actionCheckbox.value && actionSets.length > 0;
            setDropdown.enabled = enabled;
            actionDropdown.enabled = enabled;
        }
    }

    function createDocument(imageFile) {
        var srcDoc = app.open(imageFile);
        var docName = stripExtension(imageFile.name);
        var doc = srcDoc.duplicate(docName, true);
        srcDoc.close(SaveOptions.DONOTSAVECHANGES);

        app.activeDocument = doc;
        if (doc.mode !== DocumentMode.RGB) {
            doc.changeMode(ChangeMode.RGB);
        }
        doc.activeLayer = doc.layers[0];
        doc.activeLayer.name = "bg";
        return doc;
    }

    function importOverlayLayer(targetDoc, overlayFile, layerName) {
        var overlayDoc = app.open(overlayFile);
        var targetWidth = Math.round(targetDoc.width.as("px"));
        var targetHeight = Math.round(targetDoc.height.as("px"));
        if (Math.round(overlayDoc.width.as("px")) !== targetWidth ||
            Math.round(overlayDoc.height.as("px")) !== targetHeight) {
            overlayDoc.close(SaveOptions.DONOTSAVECHANGES);
            throw new Error(layerName + " 尺寸不一致");
        }

        app.activeDocument = overlayDoc;
        if (overlayDoc.mode !== DocumentMode.RGB) {
            overlayDoc.changeMode(ChangeMode.RGB);
        }
        if (overlayDoc.layers.length > 1) {
            overlayDoc.mergeVisibleLayers();
        }

        var sourceLayer = overlayDoc.activeLayer;
        var sourceBounds = getLayerBounds(sourceLayer);
        var importedLayer = sourceLayer.duplicate(targetDoc, ElementPlacement.PLACEATBEGINNING);

        app.activeDocument = targetDoc;
        setRGBChannels(targetDoc);
        targetDoc.activeLayer = importedLayer;
        importedLayer.name = layerName;
        alignLayerBounds(importedLayer, sourceBounds);
        addWhiteCornerPixels(targetDoc);

        overlayDoc.close(SaveOptions.DONOTSAVECHANGES);

        try {
            var bgLayer = targetDoc.artLayers.getByName("bg");
            importedLayer.move(bgLayer, ElementPlacement.PLACEBEFORE);
        } catch (e) {
        }
    }

    function importMaskAsAlpha(targetDoc, maskFile, channelName, skipIfAllBlack) {
        app.activeDocument = targetDoc;
        removeAlphaChannelIfExists(targetDoc, channelName);

        var maskDoc = app.open(maskFile);
        var targetWidth = Math.round(targetDoc.width.as("px"));
        var targetHeight = Math.round(targetDoc.height.as("px"));
        if (Math.round(maskDoc.width.as("px")) !== targetWidth ||
            Math.round(maskDoc.height.as("px")) !== targetHeight) {
            maskDoc.close(SaveOptions.DONOTSAVECHANGES);
            throw new Error(channelName + " 尺寸不一致");
        }

        app.activeDocument = maskDoc;
        var sourceChannel = maskDoc.channels[0];
        var alpha = sourceChannel.duplicate(targetDoc);
        maskDoc.close(SaveOptions.DONOTSAVECHANGES);

        app.activeDocument = targetDoc;
        alpha.name = channelName;
        if (skipIfAllBlack && !loadChannelSelection(targetDoc, alpha)) {
            alpha.remove();
            setRGBChannels(targetDoc);
            return false;
        }
        targetDoc.activeChannels = [alpha];
        addWhiteCornerPixels(targetDoc);
        setRGBChannels(targetDoc);
        return true;
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
        if (dx !== 0 || dy !== 0) {
            layer.translate(dx, dy);
        }
    }

    function hasChannel(doc, channelName) {
        return getChannelByName(doc, channelName) !== null;
    }

    function convertAlphaChannelToHiddenLayer(doc, channelName, layerName) {
        app.activeDocument = doc;
        var channel = getChannelByName(doc, channelName);
        if (!channel) return false;

        if (!loadChannelSelection(doc, channel)) {
            channel.remove();
            setRGBChannels(doc);
            return false;
        }

        setRGBChannels(doc);
        var layer = doc.artLayers.add();
        layer.name = layerName;
        doc.activeLayer = layer;

        var white = new SolidColor();
        white.rgb.red = 255;
        white.rgb.green = 255;
        white.rgb.blue = 255;
        doc.selection.fill(white, ColorBlendMode.NORMAL, 100, false);
        doc.selection.deselect();

        layer.visible = false;
        channel.remove();
        setRGBChannels(doc);
        return true;
    }

    function getChannelByName(doc, channelName) {
        for (var i = 0; i < doc.channels.length; i++) {
            if (doc.channels[i].name === channelName) return doc.channels[i];
        }
        return null;
    }

    function loadChannelSelection(doc, channel) {
        app.activeDocument = doc;
        doc.selection.deselect();
        doc.selection.load(channel, SelectionType.REPLACE);
        try {
            var bounds = doc.selection.bounds;
            return Math.round(bounds[2].as("px")) > Math.round(bounds[0].as("px")) &&
                Math.round(bounds[3].as("px")) > Math.round(bounds[1].as("px"));
        } catch (e) {
            doc.selection.deselect();
            return false;
        }
    }

    function getActionSets() {
        var sets = [];
        var index = 1;
        while (true) {
            var ref = new ActionReference();
            ref.putIndex(app.charIDToTypeID("ASet"), index);
            try {
                var desc = app.executeActionGet(ref);
                var name = desc.getString(app.charIDToTypeID("Nm  "));
                var count = 0;
                if (desc.hasKey(app.charIDToTypeID("NmbC"))) {
                    count = desc.getInteger(app.charIDToTypeID("NmbC"));
                }
                var actions = getActionsInSet(index, count);
                if (actions.length > 0) {
                    sets.push({
                        index: index,
                        name: name,
                        actions: actions
                    });
                }
                index++;
            } catch (e) {
                break;
            }
        }
        return sets;
    }

    function getActionsInSet(setIndex, count) {
        var actions = [];
        for (var i = 1; i <= count; i++) {
            var ref = new ActionReference();
            ref.putIndex(app.charIDToTypeID("Actn"), i);
            ref.putIndex(app.charIDToTypeID("ASet"), setIndex);
            try {
                var desc = app.executeActionGet(ref);
                actions.push({
                    index: i,
                    name: desc.getString(app.charIDToTypeID("Nm  "))
                });
            } catch (e) {
            }
        }
        return actions;
    }

    function writeReport(reportFile, imageFolder, outputRoot, total, made, settings, actionRun, actionErrors, skippedExisting, skipped) {
        reportFile.encoding = "UTF-8";
        if (!reportFile.open("w")) {
            alert("无法写入报告：\n" + reportFile.fsName);
            return;
        }
        reportFile.writeln("Create solid_inpaint PSD report");
        reportFile.writeln("Generated at: " + formatDate(new Date()));
        reportFile.writeln("Image folder: " + imageFolder.fsName);
        reportFile.writeln("ctd_inpainted folder: " + outputRoot.fsName);
        reportFile.writeln("Total image files: " + total);
        reportFile.writeln("Saved PSD files: " + made);
        reportFile.writeln("Restart: " + (settings.restart ? "yes" : "no"));
        reportFile.writeln("Run action: " + (settings.runAction ? "yes" : "no"));
        reportFile.writeln("Convert OTHER_CHANNEL to hidden layer after action: " + (settings.convertOtherChannelToLayer ? "yes" : "no"));
        if (settings.runAction) {
            reportFile.writeln("Action set: " + settings.actionSetName);
            reportFile.writeln("Action: " + settings.actionName);
        }
        reportFile.writeln("Action executed: " + actionRun.length);
        reportFile.writeln("Action failed: " + actionErrors.length);
        reportFile.writeln("Skipped existing PSD: " + skippedExisting.length);
        reportFile.writeln("Skipped or failed: " + skipped.length);
        reportFile.writeln("");

        reportFile.writeln("[ACTION_EXECUTED]");
        writeLines(reportFile, actionRun);
        reportFile.writeln("");

        reportFile.writeln("[ACTION_FAILED]");
        writeLines(reportFile, actionErrors);
        reportFile.writeln("");

        reportFile.writeln("[SKIPPED_EXISTING_PSD]");
        writeLines(reportFile, skippedExisting);
        reportFile.writeln("");

        reportFile.writeln("[SKIPPED_OR_FAILED]");
        writeLines(reportFile, skipped);
        reportFile.close();
    }

    function writeLines(file, lines) {
        if (lines.length === 0) {
            file.writeln("(none)");
            return;
        }
        for (var i = 0; i < lines.length; i++) {
            file.writeln(lines[i]);
        }
    }

    function findMaskFile(folder, stem) {
        var extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".psd"];
        for (var i = 0; i < extensions.length; i++) {
            var file = new File(folder.fsName + "/" + stem + extensions[i]);
            if (file.exists) return file;
        }
        return null;
    }

    function removeAlphaChannelIfExists(doc, channelName) {
        for (var i = doc.channels.length - 1; i >= 0; i--) {
            var channel = doc.channels[i];
            if (channel.name === channelName) {
                channel.remove();
                return;
            }
        }
    }

    function setRGBChannels(doc) {
        doc.activeChannels = [doc.channels[0], doc.channels[1], doc.channels[2]];
    }

    function addWhiteCornerPixels(doc) {
        var width = Math.round(doc.width.as("px"));
        var height = Math.round(doc.height.as("px"));
        if (width < 1 || height < 1) return;

        var white = new SolidColor();
        white.rgb.red = 255;
        white.rgb.green = 255;
        white.rgb.blue = 255;

        fillPixel(doc, 0, 0, white);
        fillPixel(doc, width - 1, 0, white);
        fillPixel(doc, 0, height - 1, white);
        fillPixel(doc, width - 1, height - 1, white);
        doc.selection.deselect();
    }

    function fillPixel(doc, x, y, color) {
        doc.selection.select([
            [x, y],
            [x + 1, y],
            [x + 1, y + 1],
            [x, y + 1]
        ]);
        doc.selection.fill(color, ColorBlendMode.NORMAL, 100, false);
    }

    function stripExtension(name) {
        return name.replace(/\.[^\.]+$/, "");
    }

    function naturalCompareNames(a, b) {
        var ax = splitNaturalName(a);
        var bx = splitNaturalName(b);
        var len = Math.min(ax.length, bx.length);
        for (var i = 0; i < len; i++) {
            var av = ax[i];
            var bv = bx[i];
            if (av[0] !== bv[0]) {
                return av[0] < bv[0] ? -1 : 1;
            }
            if (av[1] !== bv[1]) {
                return av[1] < bv[1] ? -1 : 1;
            }
            if (av.length > 2 && bv.length > 2 && av[2] !== bv[2]) {
                return av[2] < bv[2] ? -1 : 1;
            }
        }
        if (ax.length !== bx.length) {
            return ax.length < bx.length ? -1 : 1;
        }
        a = a.toLowerCase();
        b = b.toLowerCase();
        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
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
