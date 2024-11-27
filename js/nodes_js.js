import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

function populate(text) {
                    
    const v = Array.isArray(text) ? text : [text];
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

app.registerExtension({
	name: "IamMEsNodes.nodes_js",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if(!nodeData?.category?.startsWith("IamME")) {
			return;
		}
		switch (nodeData.name){
            case "AspectEmptyLatentImage":

                // TODO : this onConnectInput is probably not working, delete this after testing
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

            case "GetImageData":
                const onExec = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    const state = onExec ? onExec.apply(this, arguments):undefined

                    if (this.widgets.length > 0){
                        for (let i=0; i < this.widgets.length; i++){
                            if (this.widgets[i].name === "text2"){
                                this.widgets[i]?.onRemove?.();
                            }
                        }
                        this.widgets.length = 0
                    }

                    if (message.text) {
                        populate.call(this, message.text);
                    }
                    return state
                };
                break;

            case "LiveTextEditor":
                
                // When the node is executed we will be sent the input text, display this in the widget
                const onExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    const state = onExecuted ? onExecuted.apply(this, arguments):undefined

                    if (this.widgets.length > 3){
                        for (let i=0; i < this.widgets.length; i++){
                            if (this.widgets[i].name === "text2"){
                                this.widgets[i]?.onRemove?.();
                            }
                        }
                        this.widgets.length = 3
                    }

                    if (message.text) {
                        if (this.widgets[1].name === "modify_text"){
                            if (this.widgets[1].value !== message.text[0]){
                                this.widgets[1].value = message.text[0]
                            }
                            populate.call(this, message.text);
                        }
                    }
                    return state
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
                    const state = onExecutedLoader? onExecutedLoader.apply(this, arguments):undefined
                    
                    if (this.widgets.length > 3){
                        for (let i=0; i < this.widgets.length; i++){
                            if (this.widgets[i].name === "text2"){
                                this.widgets[i]?.onRemove?.();
                            }
                        }
                        this.widgets.length = 3
                    }

                    populate.call(this, message.text);

                    return state
                };

                break;
                
            case "ColorCorrect":
                nodeType.prototype.onNodeCreated = function () {
                    this._type = "IMAGE"
                    this.inputs_offset = nodeData.name.includes("selective")?1:0
                    this.addWidget("button", "Reset Values", null, () => {
                        const defaults = {
                            "gamma": 1,
                            "contrast": 1,
                            "exposure": 0,
                            "temperature":0.0,
                            "hue": 0,
                            "saturation": 0,
                            "value": 0,
                            "cyan_red": 0,
                            "magenta_green": 0,
                            "yellow_blue": 0
                        };
                    
                        for (const widget of this.widgets) {
                            if (widget.type !== "button" && defaults.hasOwnProperty(widget.name)) {
                                widget.value = defaults[widget.name];
                            }
                        }
                        // Force a node update if needed
                        this.onNodeChanged?.();  // or this.onPropertyChanged?.();
                    });
                }
                break;
        
            case "ConnectionBus":
                const onConnectionsChange = nodeType.prototype.onConnectionsChange;

                nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                    if (!link_info)
                        return;

                    if (index === 0){
                        if (connected){
                            this.propogateExistingTypes(link_info.node_id);
                        }
                        return ;
                    }



                    if (type==2){ //handling output slot connection
                        if (connected){
                            if (index != 0){
                                this.outputs[index].type = link_info.type;
                                this.outputs[index].label = link_info.type;
                                this.outputs[index].name = link_info.type;
                                // this.inputs[index].type = link_info.type;
                                this.inputs[index].label = link_info.type;
                                // this.inputs[index].name = link_info.type; 
                            }
                        }
                        return;
                    } 
                    else{ //handling input slot connection

                        if (connected && link_info.origin_id && index != 0){
                            const node = app.graph.getNodeById(link_info.origin_id);
                            const origin_type = node.outputs[link_info.origin_slot]?.type;
                            
                            if (origin_type && origin_type !== "*"){
                                this.outputs[index].type = origin_type;
                                this.outputs[index].label = origin_type;
                                this.outputs[index].name = origin_type;
                                // this.inputs[index].type = origin_type;
                                this.inputs[index].label = origin_type;
                                // this.inputs[index].name = origin_type; 

                                this.propogateTypeChange(index, origin_type);
                                app.graph.setDirtyCanvas(true);
                            }
                            else{
                                this.disconnectInput(index);
                            }
                            
                        }
                        return;
                    }
                    
                };

                nodeType.prototype.propogateExistingTypes = function(targetNodeId){
                    const targertNode = app.graph.getNodeById(targetNodeId);
                    if (!targertNode || targertNode.type !== "ConnectionBus") return;

                    for (let i=1; i<=10; i++){
                        if (this.outputs[i] && this.outputs[i].type && this.outputs[i].type !== "*"){
                            targertNode.outputs[i].type = this.outputs[i].type;
                            targertNode.outputs[i].label = this.outputs[i].label;
                            targertNode.outputs[i].name = this.outputs[i].name;
                            targertNode.inputs[i].label = this.outputs[i].label;

                            targertNode.propogateTypeChange(i, this.outputs[i].type);
                        }
                    }
                    app.graph.setDirtyCanvas(true);
                };



                nodeType.prototype.propogateTypeChange = function(index, new_type){
                    const connectedNodes = this.getConnectedBusNodes();

                    for (const node of connectedNodes){
                        if (node.type === "ConnectionBus"){

                            if (node.inputs[index]){
                                node.inputs[index].label = new_type;
                            }

                            if (node.outputs[index]){
                                node.outputs[index].type = new_type;
                                node.outputs[index].label = new_type;
                                node.outputs[index].name = new_type;
                            }

                            node.propogateTypeChange(index, new_type);
                        }
                    }
                };

                nodeType.prototype.getConnectedBusNodes = function(){
                    const connectedNodes = [];

                    const links = this.outputs[0].links;

                    if (links){
                        for (const linkId of links){
                            const link = app.graph.links[linkId];
                            if (link){
                                const targetNode = app.graph.getNodeById(link.target_id);
                                if (targetNode && targetNode.type === "ConnectionBus"){
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
                        return api.fetchApi('/execute', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                action: "download_model",
                                model_Name: name,
                                download_Url : url
                            })
                        })
                        .then(response => {
                            if (!response.ok) {
                                throw new Error('Download failed');
                            }
                            console.log(`Download started for model: ${name}`);
                            console.log(`message from python: ${response.message}`)
                        })
                        .catch(error => {
                            console.error(`Error downloading model try: ${error}`);
                        });
                    } catch (error) {
                        console.error(`Error downloading model: ${error}`);
                    }
                }
                
                const onModelManagerExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    const state = onModelManagerExecuted ? onModelManagerExecuted.apply(this, arguments):undefined
                    
                    // remove extra buttons
                    if (this.widgets.length > 2){
                        for (let i=2; i < this.widgets.length; i++){
                            if (this.widgets[i].type === "button"){
                                this.widgets[i]?.onRemove?.();
                            }
                        }
                        this.widgets.length = 2
                    }

                    if (message && message.names && Array.isArray(message.names)){
                        message.names.forEach((model)=>{
                            this.addWidget("button", model.name, null, () =>{
                                downloadModel(model.name, model.download_url);
                            });
                        });
                    }
                    this.setDirtyCanvas(true, true);
                    return state
                }
                break;

        }
    }
});