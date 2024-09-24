import { app } from "../../scripts/app.js";

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
            }
    }
});