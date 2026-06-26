import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ClybsChromaNodes.AdaptiveLora",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ClybAdaptiveLoraLoader" ||
            nodeData.name === "ClybAdaptiveLoraLoaderModelOnly") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                if (!this.all_widgets) {
                    this.all_widgets = [...this.widgets];
                }

                this.updateWidgets = function() {
                    let lastActiveIdx = 1;
                    for (let i = 1; i <= 20; i++) {
                        const w = this.all_widgets.find(w => w.name === `lora_name_${i}`);
                        if (w && w.value && w.value !== "none") {
                            lastActiveIdx = i + 1;
                        }
                    }
                    if (lastActiveIdx > 20) lastActiveIdx = 20;

                    const newWidgets = [];
                    for (let w of this.all_widgets) {
                        const name = w.name;
                        if (name.startsWith("lora_name_") || name.startsWith("strength_model_") || name.startsWith("strength_clip_")) {
                            const idx = parseInt(name.split("_").pop());
                            if (idx <= lastActiveIdx) {
                                newWidgets.push(w);
                            }
                        } else {
                            newWidgets.push(w);
                        }
                    }
                    this.widgets = newWidgets;
                    this.computeSize();
                    app.graph.setDirtyCanvas(true, true);
                };

                for (let w of this.all_widgets) {
                    if (w.name.startsWith("lora_name_")) {
                        const oldCallback = w.callback;
                        w.callback = (v) => {
                            if (oldCallback) oldCallback.apply(this, [v]);
                            this.updateWidgets();
                        };
                    }
                }

                this.updateWidgets();
                return r;
            };

            // Ensure all widgets are serialized even if hidden
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                if (onSerialize) onSerialize.apply(this, arguments);
                if (this.all_widgets) {
                    o.widgets_values = this.all_widgets.map(w => w.value);
                }
            };

            // Ensure we can load all widgets correctly
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(o) {
                if (this.all_widgets && o.widgets_values) {
                    // Pre-fill all_widgets with loaded values before updateWidgets runs
                    for (let i = 0; i < this.all_widgets.length; i++) {
                        if (o.widgets_values[i] !== undefined) {
                            this.all_widgets[i].value = o.widgets_values[i];
                        }
                    }
                }
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                if (this.updateWidgets) this.updateWidgets();
                return r;
            };
        }
    }
});
