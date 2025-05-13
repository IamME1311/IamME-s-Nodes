import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

function populate(text) {
  const v = Array.isArray(text) ? text : [text];
  if (!v[0]) {
    v.shift();
  }
  for (const list of v) {
    const w = ComfyWidgets["STRING"](
      this,
      "text2",
      ["STRING", { multiline: true }],
      app
    ).widget;
    w.inputEl.readOnly = true;
    w.inputEl.style.opacity = 0.6;
    w.value = list;
  }

  requestAnimationFrame(() => {
    const sz = this.computeSize();
    if (sz[0] < this.size[0]) {
      sz[0] = this.size[0];
    }
    if (sz[1] < this.size[1]) {
      sz[1] = this.size[1];
    }
    this.onResize?.(sz);
    app.graph.setDirtyCanvas(true, false);
  });
}

/**
 * Default and limit configurations
 */
const CONFIG = {
    DEFAULT_VERSION_AMOUNT: 5,
    MAX_VERSION_AMOUNT: 100,
    MIN_VERSION_AMOUNT: 1
};

/**
 * Visual style configuration for version tabs
 */
const TAB_STYLE = {
    width: 40,
    height: 18,
    fontSize: 10,
    normalColor: "#333333",
    selectedColor: "#666666",
    textColor: "white",
    borderRadius: 4,
    spacing: 10,
    offset: 10,
    yPosition: 10
};

function createTabConfig(numVersions) {
    return {
        ...TAB_STYLE,
        labels: Array.from({length: numVersions}, (_, i) => (i + 1).toString())
    };
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

app.registerExtension({
  name: "IamMEsNodes.nodes_js",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    
    if (!nodeData?.category?.startsWith("IamME")) {

      return;
    }
    switch (nodeData.name) {
      case "AspectEmptyLatentImage":
        // TODO : this onConnectInput is probably not working, delete this after testing
        const onAspectLatentImageConnectInput =
          nodeType.prototype.onConnectInput;
        nodeType.prototype.onConnectInput = function (
          targetSlot,
          type,
          output,
          originNode,
          originSlot
        ) {
          const v = onAspectLatentImageConnectInput
            ? onAspectLatentImageConnectInput.apply(this, arguments)
            : undefined;
          this.outputs[1]["name"] = "width";
          this.outputs[2]["name"] = "height";
          return v;
        };
        const onAspectLatentImageExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
          const r = onAspectLatentImageExecuted
            ? onAspectLatentImageExecuted.apply(this, arguments)
            : undefined;
          let values = message["text"].toString().split("x").map(Number);
          this.outputs[1]["name"] = values[1] + " width";
          this.outputs[2]["name"] = values[0] + " height";
          return r;
        };
        break;

      case "GetImageData":
        const onExec = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
          const state = onExec ? onExec.apply(this, arguments) : undefined;

          if (this.widgets.length > 0) {
            for (let i = 0; i < this.widgets.length; i++) {
              if (this.widgets[i].name === "text2") {
                this.widgets[i]?.onRemove?.();
              }
            }
            this.widgets.length = 0;
          }

          if (message.text) {
            populate.call(this, message.text);
          }
          return state;
        };
        break;

      case "LiveTextEditor":
        // When the node is executed we will be sent the input text, display this in the widget
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
          const state = onExecuted
            ? onExecuted.apply(this, arguments)
            : undefined;

          if (this.widgets.length > 3) {
            for (let i = 0; i < this.widgets.length; i++) {
              if (this.widgets[i].name === "text2") {
                this.widgets[i]?.onRemove?.();
              }
            }
            this.widgets.length = 3;
          }

          if (message.text) {
            if (this.widgets[1].name === "modify_text") {
              if (this.widgets[1].value !== message.text[0]) {
                this.widgets[1].value = message.text[0];
              }
              populate.call(this, message.text);
            }
          }
          return state;
        };

        // this will only trigger when page is reloaded, probably don't need this.
        // const onConfigure = nodeType.prototype.onConfigure;

        // nodeType.prototype.onConfigure = function (){
        //     console.log("inside on configure");
        //     console.log(this.widgets);
        // };
        break;

      case "ImageBatchLoader":
        // When the node is executed we will be sent the input text, display this in the widget
        const onExecutedLoader = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
          const state = onExecutedLoader
            ? onExecutedLoader.apply(this, arguments)
            : undefined;

          if (this.widgets.length > 3) {
            for (let i = 0; i < this.widgets.length; i++) {
              if (this.widgets[i].name === "text2") {
                this.widgets[i]?.onRemove?.();
              }
            }
            this.widgets.length = 3;
          }

          populate.call(this, message.text);

          return state;
        };

        break;

      case "ColorCorrect":
        nodeType.prototype.onNodeCreated = function () {
          this._type = "IMAGE";
          this.inputs_offset = nodeData.name.includes("selective") ? 1 : 0;
          this.addWidget("button", "Reset Values", null, () => {
            const defaults = {
              gamma: 1,
              contrast: 1,
              exposure: 0,
              temperature: 0.0,
              hue: 0,
              saturation: 0,
              value: 0,
              cyan_red: 0,
              magenta_green: 0,
              yellow_blue: 0,
            };

            for (const widget of this.widgets) {
              if (
                widget.type !== "button" &&
                defaults.hasOwnProperty(widget.name)
              ) {
                widget.value = defaults[widget.name];
              }
            }
            // Force a node update if needed
            this.onNodeChanged?.(); // or this.onPropertyChanged?.();
          });
        };
        break;

      case "ConnectionBus":
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;

        nodeType.prototype.onConnectionsChange = function (
          type,
          index,
          connected,
          link_info
        ) {
          if (!link_info) return;

          if (index === 0) {
            if (connected) {
              this.propogateExistingTypes(link_info.node_id);
            }
            return;
          }

          if (type == 2) {
            //handling output slot connection
            if (connected) {
              if (index != 0) {
                this.outputs[index].type = link_info.type;
                this.outputs[index].label = link_info.type;
                this.outputs[index].name = link_info.type;
                // this.inputs[index].type = link_info.type;
                this.inputs[index].label = link_info.type;
                // this.inputs[index].name = link_info.type;
              }
            }
            return;
          } else {
            //handling input slot connection

            if (connected && link_info.origin_id && index != 0) {
              const node = app.graph.getNodeById(link_info.origin_id);
              const origin_type = node.outputs[link_info.origin_slot]?.type;

              if (origin_type && origin_type !== "*") {
                this.outputs[index].type = origin_type;
                this.outputs[index].label = origin_type;
                this.outputs[index].name = origin_type;
                // this.inputs[index].type = origin_type;
                this.inputs[index].label = origin_type;
                // this.inputs[index].name = origin_type;

                this.propogateTypeChange(index, origin_type);
                app.graph.setDirtyCanvas(true);
              } else {
                this.disconnectInput(index);
              }
            }
            return;
          }
        };

        nodeType.prototype.propogateExistingTypes = function (targetNodeId) {
          const targertNode = app.graph.getNodeById(targetNodeId);
          if (!targertNode || targertNode.type !== "ConnectionBus") return;

          for (let i = 1; i <= 10; i++) {
            if (
              this.outputs[i] &&
              this.outputs[i].type &&
              this.outputs[i].type !== "*"
            ) {
              targertNode.outputs[i].type = this.outputs[i].type;
              targertNode.outputs[i].label = this.outputs[i].label;
              targertNode.outputs[i].name = this.outputs[i].name;
              targertNode.inputs[i].label = this.outputs[i].label;

              targertNode.propogateTypeChange(i, this.outputs[i].type);
            }
          }
          app.graph.setDirtyCanvas(true);
        };

        nodeType.prototype.propogateTypeChange = function (index, new_type) {
          const connectedNodes = this.getConnectedBusNodes();

          for (const node of connectedNodes) {
            if (node.type === "ConnectionBus") {
              if (node.inputs[index]) {
                node.inputs[index].label = new_type;
              }

              if (node.outputs[index]) {
                node.outputs[index].type = new_type;
                node.outputs[index].label = new_type;
                node.outputs[index].name = new_type;
              }

              node.propogateTypeChange(index, new_type);
            }
          }
        };

        nodeType.prototype.getConnectedBusNodes = function () {
          const connectedNodes = [];

          const links = this.outputs[0].links;

          if (links) {
            for (const linkId of links) {
              const link = app.graph.links[linkId];
              if (link) {
                const targetNode = app.graph.getNodeById(link.target_id);
                if (targetNode && targetNode.type === "ConnectionBus") {
                  connectedNodes.push(targetNode);
                }
              }
            }
          }
          return connectedNodes;
        };
        break;

      case "ModelManager":
        function downloadModel(name, url) {
          try {
            // Construct the API call to ComfyUI's server
            return api
              .fetchApi("/execute", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({
                  action: "download_model",
                  model_Name: name,
                  download_Url: url,
                }),
              })
              .then((response) => {
                if (!response.ok) {
                  throw new Error("Download failed");
                }
                console.log(`Download started for model: ${name}`);
                console.log(`message from python: ${response.message}`);
              })
              .catch((error) => {
                console.error(`Error downloading model try: ${error}`);
              });
          } catch (error) {
            console.error(`Error downloading model: ${error}`);
          }
        }

        const onModelManagerExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
          const state = onModelManagerExecuted
            ? onModelManagerExecuted.apply(this, arguments)
            : undefined;

          // remove extra buttons
          if (this.widgets.length > 2) {
            for (let i = 2; i < this.widgets.length; i++) {
              if (this.widgets[i].type === "button") {
                this.widgets[i]?.onRemove?.();
              }
            }
            this.widgets.length = 2;
          }

          if (message && message.names && Array.isArray(message.names)) {
            message.names.forEach((model) => {
              this.addWidget("button", model.name, null, () => {
                downloadModel(model.name, model.download_url);
              });
            });
          }
          this.setDirtyCanvas(true, true);
          return state;
        };
        break;

      case "TextBox":
        const node = nodeType.prototype;
        // Store original methods
        const onNodeCreated = node.onNodeCreated;
        const onDrawBackground = node.onDrawBackground;
        const getBounding = node.getBounding;
        const onSerialize = node.onSerialize;
        const onConfigure = node.onConfigure;

        // Add onNodeCreated to the node
        node.onNodeCreated = function () {
          if (onNodeCreated) {
            onNodeCreated.apply(this, arguments);
          }

          // Add properties
          this.addProperty(
            "versionAmount",
            CONFIG.DEFAULT_VERSION_AMOUNT,
            "number"
          );

          // Initialize node
          if (!this.properties) {
            this.properties = {};
          }
          this.properties.versionAmount =
            this.properties.versionAmount || CONFIG.DEFAULT_VERSION_AMOUNT;

          // Create tab config based on number of versions
          this.tabConfig = createTabConfig(this.properties.versionAmount);

          // Initialize tab state and content
          this.activeTab = 0;
          this.tabContents = Array(this.properties.versionAmount).fill("");

          // Store reference to the text widget
          this.textWidget = this.widgets.find((w) => w.name === "text");

          // Initial content setup
          if (this.textWidget) {
            this.textWidget.value = this.tabContents[this.activeTab];

            // Add debounced change event listener
            if (this.textWidget.inputEl) {
              const saveContent = debounce(() => {
                this.tabContents[this.activeTab] = this.textWidget.value;
              }, 300);

              this.textWidget.inputEl.addEventListener("input", saveContent);
            }
          }
        };

        node.onPropertyChanged = function (name, value) {
          if (name === "versionAmount") {
            const newValue = Math.max(
              CONFIG.MIN_VERSION_AMOUNT,
              Math.min(CONFIG.MAX_VERSION_AMOUNT, Math.floor(value))
            );
            const oldContents = [...this.tabContents];
            this.properties.versionAmount = newValue;
            this.tabConfig = createTabConfig(newValue);
            this.tabContents = Array(newValue)
              .fill("")
              .map((_, i) => oldContents[i] || "");
            this.activeTab = Math.min(this.activeTab, newValue - 1);
            if (this.textWidget) {
              this.textWidget.value = this.tabContents[this.activeTab];
            }
            this.setDirtyCanvas(true);
          }
        };

        node.onDrawBackground = function (ctx) {
          if (onDrawBackground) {
            onDrawBackground.apply(this, arguments);
          }

          if (this.flags.collapsed) return;

          ctx.save();

          // Create clipping region for overflow
          const nodeWidth = this.size[0];
          const clipPadding = 10;
          ctx.beginPath();
          ctx.rect(
            clipPadding,
            this.tabConfig.yPosition - 5,
            nodeWidth - 2 * clipPadding,
            this.tabConfig.height + 10
          );
          ctx.clip();

          // Draw tabs
          this.tabConfig.labels.forEach((label, i) => {
            const x =
              this.tabConfig.offset +
              (this.tabConfig.width + this.tabConfig.spacing) * i;
            const y = this.tabConfig.yPosition;

            ctx.fillStyle =
              i === this.activeTab
                ? this.tabConfig.selectedColor
                : this.tabConfig.normalColor;
            ctx.beginPath();
            ctx.roundRect(
              x,
              y,
              this.tabConfig.width,
              this.tabConfig.height,
              this.tabConfig.borderRadius
            );
            ctx.fill();

            ctx.fillStyle = this.tabConfig.textColor;
            ctx.font = `${this.tabConfig.fontSize}px Arial`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(
              label,
              x + this.tabConfig.width / 2,
              y + this.tabConfig.height / 2
            );
          });

          ctx.restore();
        };

        node.onMouseDown = function (event, local_pos, graphCanvas) {
          const [x, y] = local_pos;
          const { yPosition, height, width, spacing, offset, labels } =
            this.tabConfig;

          if (y >= yPosition && y <= yPosition + height) {
            for (let i = 0; i < labels.length; i++) {
              const tabX = offset + (width + spacing) * i;
              if (x >= tabX && x <= tabX + width) {
                if (i === this.activeTab) return false;

                if (this.textWidget) {
                  this.tabContents[this.activeTab] = this.textWidget.value;
                }

                this.activeTab = i;

                if (this.textWidget) {
                  this.textWidget.value = this.tabContents[i];
                }

                this.setDirtyCanvas(true);
                return true;
              }
            }
          }

          return false;
        };

        node.getBounding = function () {
          const bounds =
            getBounding?.apply(this, arguments) || new Float32Array(4);
          const tabsHeight =
            Math.abs(this.tabConfig.yPosition) + this.tabConfig.height;
          bounds[1] -= tabsHeight;
          bounds[3] += tabsHeight;
          return bounds;
        };

        node.onSerialize = function (o) {
          if (onSerialize) {
            onSerialize.apply(this, arguments);
          }
          o.tabContents = this.tabContents;
          o.activeTab = this.activeTab;
        };

        node.onConfigure = function (o) {
          if (onConfigure) {
            onConfigure.apply(this, arguments);
          }

          this.tabConfig = createTabConfig(this.properties.versionAmount);

          if (o.tabContents && Array.isArray(o.tabContents)) {
            this.tabContents = Array(this.properties.versionAmount)
              .fill("")
              .map((_, i) => o.tabContents[i] || "");
            this.activeTab =
              o.activeTab >= 0 && o.activeTab < this.properties.versionAmount
                ? o.activeTab
                : 0;
            if (this.textWidget) {
              this.textWidget.value = this.tabContents[this.activeTab];
            }
          }
        };
        break;
    }
  },
});
