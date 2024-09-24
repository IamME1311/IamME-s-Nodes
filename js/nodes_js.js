import { app } from "../../scripts/app.js";
import { recursiveLinkUpstream } from "./utils.js"

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

            case "ImageLivePreview":
                const onImageLivePreviewConnectInput = nodeType.prototype.onConnectInput;
                const inputList = (index !== null) ? [index] : [...Array(node.inputs.length).keys()]
                if (inputList.length === 0) { return }
        
                for (let i of inputList) {
                    const connectedNodes = recursiveLinkUpstream(node, node.inputs[i].type, 0, i)
                    
                    if (connectedNodes.length !== 0) {
                        for (let [node_ID, depth] of connectedNodes) {
                            const connectedNode = node.graph._nodes_by_id[node_ID]
        
                            if (connectedNode.type !== "ImageLivePreview") {
        
                                const [endWidth, endHeight] = getSizeFromNode(connectedNode)
        
                                if (endWidth && endHeight) {
                                    if (i === 0) {
                                        node.sampleToID = connectedNode.id
                                    } else {
                                        node.properties["values"][i-1][3] = connectedNode.id
                                    }
                                    break
                                }
                            }
                        }
                    }
                }
        }
    }
});