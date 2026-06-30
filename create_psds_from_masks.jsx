/*
Create PSD files from original images and same-name mask images.

Run in Photoshop:
File > Scripts > Browse... > create_psds_from_masks.jsx

Inputs selected in one dialog:
- image folder
- mask folder
- output PSD folder name
- Photoshop action set/action

Output:
<image folder>/<output folder name>/<name>.psd

Each PSD contains:
- bg
- alpha channel: OTHER_CHANNEL

The selected Photoshop action is executed only when the source mask has content.
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
        var maskFolder = new Folder(settings.maskFolder);
        var psdFolder = new Folder(imageFolder.fsName + "/" + settings.outputFolderName);

        if (!imageFolder.exists) {
            alert("原图文件夹不存在：\n" + imageFolder.fsName);
            return;
        }
        if (!maskFolder.exists) {
            alert("mask 图文件夹不存在：\n" + maskFolder.fsName);
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
        var actionSkipped = [];
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

            var maskFile = findMaskFile(maskFolder, stem);

            if (!maskFile) {
                skipped.push(imageFile.name + "：缺少 mask");
                continue;
            }

            var doc = null;
            try {
                doc = createDocument(imageFile);
                var maskHasContent = importMaskAsAlpha(doc, maskFile, "OTHER_CHANNEL");

                if (maskHasContent) {
                    try {
                        app.activeDocument = doc;
                        setRGBChannels(doc);
                        app.doAction(settings.actionName, settings.actionSetName);
                        actionRun.push(imageFile.name);
                    } catch (actionErr) {
                        actionErrors.push(imageFile.name + "：" + actionErr.message);
                    }
                } else {
                    actionSkipped.push(imageFile.name + "：mask 为空");
                }

                app.activeDocument = doc;
                setRGBChannels(doc);
                var saveOptions = new PhotoshopSaveOptions();
                saveOptions.alphaChannels = true;
                saveOptions.layers = true;
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
            new File(psdFolder.fsName + "/create_psds_from_masks_report.txt"),
            imageFolder,
            maskFolder,
            imageFiles.length,
            made,
            settings,
            actionRun,
            actionSkipped,
            actionErrors,
            skippedExisting,
            skipped
        );

        var message = "PSD 生成完成：" + made + " 个\n输出目录：\n" + psdFolder.fsName;
        message += "\n执行动作：" + actionRun.length;
        message += "\nmask 为空未执行：" + actionSkipped.length;
        message += "\n动作失败：" + actionErrors.length;
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
        alert("Create mask PSDs failed:\n" + e.toString() + "\nLine: " + (e.line || "unknown"));
    } finally {
        app.preferences.rulerUnits = oldRulerUnits;
    }

    function showSettingsDialog() {
        var actionSets = getActionSets();
        var dialog = new Window("dialog", "生成 mask PSD");
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

        var maskGroup = dialog.add("group");
        maskGroup.orientation = "row";
        maskGroup.alignChildren = ["fill", "center"];
        maskGroup.add("statictext", undefined, "mask 图文件夹：");
        var maskPathInput = maskGroup.add("edittext", undefined, "");
        maskPathInput.characters = 52;
        var maskBrowseButton = maskGroup.add("button", undefined, "选择");

        var outputGroup = dialog.add("group");
        outputGroup.orientation = "row";
        outputGroup.alignChildren = ["left", "center"];
        outputGroup.add("statictext", undefined, "输出 PSD 文件夹名：");
        var outputNameInput = outputGroup.add("edittext", undefined, "psd");
        outputNameInput.characters = 24;

        var restartGroup = dialog.add("group");
        restartGroup.orientation = "row";
        restartGroup.alignChildren = ["left", "center"];
        var restartCheckbox = restartGroup.add("checkbox", undefined, "重新开始（覆盖已有 PSD）");
        restartCheckbox.value = false;

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
            }
        };

        maskBrowseButton.onClick = function () {
            var selected = Folder.selectDialog("选择 mask 图文件夹");
            if (selected) {
                maskPathInput.text = selected.fsName;
            }
        };

        setDropdown.onChange = function () {
            refreshActionDropdown();
        };

        okButton.onClick = function () {
            if (!trimString(imagePathInput.text)) {
                alert("请选择原图文件夹。");
                return;
            }
            if (!trimString(maskPathInput.text)) {
                alert("请选择 mask 图文件夹。");
                return;
            }
            if (!isValidFolderName(trimString(outputNameInput.text))) {
                alert("请输入有效的输出 PSD 文件夹名，不能包含 / \\ : * ? \" < > |。");
                return;
            }
            if (!setDropdown.selection || !actionDropdown.selection) {
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
            maskFolder: trimString(maskPathInput.text),
            outputFolderName: trimString(outputNameInput.text),
            restart: restartCheckbox.value,
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
            var enabled = actionSets.length > 0;
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

    function importMaskAsAlpha(targetDoc, maskFile, channelName) {
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
        var hasContent = !isChannelAllBlack(sourceChannel);
        var alpha = sourceChannel.duplicate(targetDoc);
        maskDoc.close(SaveOptions.DONOTSAVECHANGES);

        app.activeDocument = targetDoc;
        alpha.name = channelName;
        setRGBChannels(targetDoc);
        return hasContent;
    }

    function isChannelAllBlack(channel) {
        var histogram = channel.histogram;
        for (var i = 1; i < histogram.length; i++) {
            if (histogram[i] > 0) return false;
        }
        return true;
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

    function writeReport(reportFile, imageFolder, maskFolder, total, made, settings, actionRun, actionSkipped, actionErrors, skippedExisting, skipped) {
        reportFile.encoding = "UTF-8";
        if (!reportFile.open("w")) {
            alert("无法写入报告：\n" + reportFile.fsName);
            return;
        }
        reportFile.writeln("Create mask PSD report");
        reportFile.writeln("Generated at: " + formatDate(new Date()));
        reportFile.writeln("Image folder: " + imageFolder.fsName);
        reportFile.writeln("Mask folder: " + maskFolder.fsName);
        reportFile.writeln("Output folder name: " + settings.outputFolderName);
        reportFile.writeln("Total image files: " + total);
        reportFile.writeln("Saved PSD files: " + made);
        reportFile.writeln("Action set: " + settings.actionSetName);
        reportFile.writeln("Action: " + settings.actionName);
        reportFile.writeln("Restart: " + (settings.restart ? "yes" : "no"));
        reportFile.writeln("Action executed: " + actionRun.length);
        reportFile.writeln("Action skipped empty mask: " + actionSkipped.length);
        reportFile.writeln("Action failed: " + actionErrors.length);
        reportFile.writeln("Skipped existing PSD: " + skippedExisting.length);
        reportFile.writeln("Skipped or failed: " + skipped.length);
        reportFile.writeln("");

        reportFile.writeln("[ACTION_EXECUTED]");
        writeLines(reportFile, actionRun);
        reportFile.writeln("");

        reportFile.writeln("[ACTION_SKIPPED_EMPTY_MASK]");
        writeLines(reportFile, actionSkipped);
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

    function isValidFolderName(value) {
        if (!value) return false;
        return !/[\/\\:\*\?"<>\|]/.test(value);
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
