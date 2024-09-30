import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";


// function recursiveLinkUpstream(node, type, depth, index=null) {
// 	depth += 1
// 	let connections = []
// 	const inputList = (index !== null) ? [index] : [...Array(node.inputs.length).keys()]
// 	if (inputList.length === 0) { return }
// 	for (let i of inputList) {
// 		const link = node.inputs[i].link
// 		if (link) {
// 			const nodeID = node.graph.links[link].origin_id
// 			const slotID = node.graph.links[link].origin_slot
// 			const connectedNode = node.graph._nodes_by_id[nodeID]

// 			if (connectedNode.outputs[slotID].type === type) {

// 				connections.push([connectedNode.id, depth])

// 				if (connectedNode.inputs) {
// 					const index = (connectedNode.type === "LatentComposite") ? 0 : null
// 					connections = connections.concat(recursiveLinkUpstream(connectedNode, type, depth, index))
// 				} else {
					
// 				}
// 			}
// 		}
// 	}
// 	return connections
// }

app.registerExtension({
	name: "IamMEsNodes.nodes_js",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if(!nodeData?.category?.startsWith("IamME")) {
			return;
		}
		switch (nodeData.name){
            case "AspectEmptyLatentImage":
                const onAspectLatentImageConnectInput = nodeType.prototype.onConnectInput;
                nodeType.prototype.onConnectInput = function (targetSlot, type, output, originNode, originSlot) {
                    const v = onAspectLatentImageConnectInput? onAspectLatentImageConnectInput.apply(this, arguments): undefined
                    this.outputs[1]["name"] = "width"
                    this.outputs[2]["name"] = "height" 
                    return v;
                }
                const onAspectLatentImageExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function(message) {
                    const r = onAspectLatentImageExecuted? onAspectLatentImageExecuted.apply(this,arguments): undefined
                    let values = message["text"].toString().split('x').map(Number);
                    this.outputs[1]["name"] = values[1] + " width"
                    this.outputs[2]["name"] = values[0] + " height"  
                    return r
                }
                break;


            case "LiveTextEditor":
            case "TriggerWordProcessor":
                function populate(text) {
                    if (this.widgets) {
                        for (let i = 2; i < this.widgets.length; i++) {
                            this.widgets[i].onRemove?.();
                        }
                        this.widgets.length = 2;
                    }
                    
                    const v = [...text];
                    if (!v[0]) {
                        v.shift();
                    }
                    for (const list of v) {
                        const w = ComfyWidgets["STRING"](this, "text2", ["STRING", { multiline: true }], app).widget;
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
    
                // When the node is executed we will be sent the input text, display this in the widget
                const onExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    onExecuted?.apply(this, arguments);
                    populate.call(this, message.text);
                };
    
                const onConfigure = nodeType.prototype.onConfigure;
                nodeType.prototype.onConfigure = function () {
                    onConfigure?.apply(this, arguments);
                    if (this.widgets_values?.length) {
                        populate.call(this, this.widgets_values.slice(+this.widgets_values.length > 1));
                    }
                };
                       
            
        }
    }
});