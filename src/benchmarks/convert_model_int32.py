import onnx
from onnx import numpy_helper, helper

def force_int32_everywhere(model_path, out_path):
    model = onnx.load(model_path)
    graph = model.graph

    # 1. Convert Initializers (Weights/Constants)
    for tensor in graph.initializer:
        if tensor.data_type == onnx.TensorProto.INT64:
            arr = numpy_helper.to_array(tensor)
            # Ensure no overflow before casting
            new_tensor = numpy_helper.from_array(arr.astype('int32'), name=tensor.name)
            tensor.CopyFrom(new_tensor)

    # 2. Convert Value Info (Internal tensor shapes/types)
    for value_info in list(graph.value_info) + list(graph.input) + list(graph.output):
        if value_info.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            value_info.type.tensor_type.elem_type = onnx.TensorProto.INT32

    # 3. Handle Constant Nodes (Hidden INT64s)
    for node in graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.t.data_type == onnx.TensorProto.INT64:
                    arr = numpy_helper.to_array(attr.t)
                    new_t = numpy_helper.from_array(arr.astype('int32'))
                    attr.t.CopyFrom(new_t)
        
        # 4. Handle Cast Nodes (Change target type from int64 to int32)
        if node.op_type == 'Cast':
            for attr in node.attribute:
                if attr.name == 'to' and attr.i == onnx.TensorProto.INT64:
                    attr.i = onnx.TensorProto.INT32

    onnx.save(model, out_path)
    print(f"✅ Deep conversion complete: {out_path}")

force_int32_everywhere("models/face_detection/scrfd10gkps.onnx", "models/face_detection/scrfd10gkps_fixed_int32.onnx")