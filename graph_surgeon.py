import onnx_graphsurgeon as gs
import onnx
import numpy as np 

input_model_path = "data/ssd/ssd-mobilenet.onnx"
output_model_path = "data/ssd/model_gs.onnx"

@gs.Graph.register()
def trt_batched_nms(self, boxes_input, scores_input, nms_output,
                    share_location, num_classes):

    boxes_input.outputs.clear()
    scores_input.outputs.clear()
    nms_output.inputs.clear()

    attrs = {
        "shareLocation": share_location,
        "numClasses": num_classes,
        "backgroundLabelId": 0,
        "topK": 100,
        "keepTopK": 100,
        "scoreThreshold": 0.3,
        "iouThreshold": 0.6,
        "isNormalized": True,
        "clipBoxes": True
    }
    return self.layer(op="BatchedNMS_TRT", attrs=attrs,
                      inputs=[boxes_input, scores_input],
                      outputs=[nms_output])

graph = gs.import_onnx(onnx.load(input_model_path))
# graph.inputs[0].shape=[1,3,300,300]
# print(graph.inputs[0].shape)


nums_out = gs.Variable("nms_out", dtype=np.float32)
nums_out.shape = []
graph.outputs = [nums_out]

tmap = graph.tensors()
graph.trt_batched_nms(tmap["boxes"], tmap["scores"], 
                      tmap["nms_out"], share_location=False, 
                      num_classes=2)

graph.cleanup().toposort()

onnx.checker.check_model(gs.export_onnx(graph))
onnx.save_model(gs.export_onnx(graph), output_model_path)
print("Saving the ONNX model to {}".format(output_model_path))
