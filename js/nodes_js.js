import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";


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

            case "GetImageData":
            case "LiveTextEditor":
                function populate(text) {
                    const isGetImageData = nodeData.name === "GetImageData";
                    if (this.widgets) {
                        if (isGetImageData){
                            const pos = this.widgets.findIndex((w) => w.name === 'text2');
                            if (pos !== -1) {
                                for (let i = pos; i < this.widgets.length; i++) {
                                    this.widgets[i]?.onRemove?.();
                                }
                                this.widgets.length = pos;
                            }
                        }
                        else{
                            for (let i = 2; i < this.widgets.length; i++) {
                                this.widgets[i].onRemove?.();
                            }
                            this.widgets.length = 2;
                        }
                    }
                    
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
    
                // When the node is executed we will be sent the input text, display this in the widget
                const onExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    onExecuted?.apply(this, arguments);
                    if (message.text) {
                        populate.call(this, message.text);
                    }
                };
    
                const onConfigure = nodeType.prototype.onConfigure;
                nodeType.prototype.onConfigure = function () {
                    onConfigure?.apply(this, arguments);
                    if (this.widgets_values?.length) {
                        const values = this.widgets_values.slice(0, 2);
                        populate.call(this, values);
                        this.widgets_values = values;
                    }
                };
                break;
                       
            case "ImageBatchLoader":
                function populatedata(text) {
                    if (this.widgets) {
                        for (let i = 3; i < this.widgets.length; i++) {
                            this.widgets[i].onRemove?.();
                        }
                        this.widgets.length = 3;
                    }
                    
                    const v = [...text];
                    if (!v[0]) {
                        v.shift();
                    }
                    for (const list of v) {
                        const w = ComfyWidgets["STRING"](this, "text_new", ["STRING", { multiline: true }], app).widget;
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
                const onExecutedLoader = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    onExecutedLoader?.apply(this, arguments);
                    populatedata.call(this, message.text);
                };
    
                const onConfigureLoader = nodeType.prototype.onConfigure;
                nodeType.prototype.onConfigure = function () {
                    onConfigureLoader?.apply(this, arguments);
                    if (this.widgets_values?.length) {
                        populatedata.call(this, this.widgets_values.slice(+this.widgets_values.length > 1));
                    }
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
                function downloadModel(modelName, downloadUrl) {
                    try {
                        // Get the node's ID from the graph
                        const nodeId = this.id;
                        console.log("downloadModel called");
                        // Construct the API call to ComfyUI's server
                        return api.fetchApi('/execute', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                node_id: nodeId,
                                action: "download_model",
                                model_Name: modelName,
                                download_Url : downloadUrl
                            })
                        })
                        .then(response => {
                            if (!response.ok) {
                                throw new Error('Download failed');
                            }
                            console.log(`Download started for model: ${modelName}`);
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
                    console.log("ModelManager onExecuted called", message);
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
                        console.log("Creating buttons for models:", message.names);
                        message.names.forEach((model)=>{
                            this.addWidget("button", model.name, null, () =>{
                                console.log(`Button clicked for model: ${model.name}`);
                                downloadModel.call(model.name, model.download_url);
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