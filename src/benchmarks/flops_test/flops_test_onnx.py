import onnx
import onnxruntime as ort
import os
import numpy as np
import onnx_tool
from onnx_tool import create_ndarray_f32 #or use numpy.ones(shape,numpy.float32) is ok
from onnx import StringStringEntryProto
from typing import Tuple
from utils.utils import generate_md5_hash, compute_mem_size_units, compute_magnitude_units

# These ops have no computation
NoMacsOps = (
    'Identity', 'Constant', 'Shape', 'Squeeze', 'Unsqueeze', 'Reshape', 'ConstantOfShape', 'Cast', 'Pad', 'Concat',
    'Slice', 'Gather'
)

def model_profile(m, dynamic_shapes: {str: tuple} = None,
            hidden_ops: [str] = NoMacsOps, mcfg={'verbose': False}, save_profile: str = None,
            save_model: str = None, shape_only:bool=False, no_shape:bool=False) -> None:
    model = onnx_tool.loadmodel(m, mcfg)
    g = model.graph
    gtmr = onnx_tool.timer()
    g.graph_reorder_nodes()
    gtmr.start()
    #print(f'profile model {m} with dynamic shapes {dynamic_shapes}')
    g.shape_infer(dynamic_shapes)
    g.log(f'infered all tensor shapes, time cost {gtmr.stop():.3f} s')
    gtmr.start()
    g.profile()
    g.log(f'profile all nodes, time cost {gtmr.stop():.3f} s')
    #g.print_node_map(save_profile, exclude_ops=hidden_ops)
    if save_model is not None:
        model.save_model(save_model, shape_only=shape_only, no_shape=no_shape)
    return g.macs


def get_model_size(onnx_model_path: str) -> str:
    """Get the size of the ONNX model file in bytes and convert to megabytes."""
    size_bytes = os.path.getsize(onnx_model_path)
    size = compute_mem_size_units(size_bytes)
    return size

def get_input_output_dims(onnx_model_path: str) -> Tuple[dict, dict]:
    """Extract input and output dimensions of the ONNX model using onnxruntime."""
    session = ort.InferenceSession(onnx_model_path)

    # Extract input dimensions
    input_dims = {}
    for input_meta in session.get_inputs():
        input_dims[input_meta.name] = input_meta.shape

    # Extract output dimensions
    output_dims = {}
    for output_meta in session.get_outputs():
        output_dims[output_meta.name] = output_meta.shape
    return input_dims, output_dims

def count_model_params(onnx_model_path: str) -> str:
    """Count the number of parameters (weights and biases) in the ONNX model."""
    onnx_model = onnx.load(onnx_model_path)
    params = 0

    # Loop through the initializers (model weights)
    for initializer in onnx_model.graph.initializer:
        param_shape = list(initializer.dims)
        param_size = np.prod(param_shape)  # Total parameters for this layer
        params += param_size
    params = compute_magnitude_units(params)
    return params

def old_estimate_flops(onnx_model_path: str) -> str:
    """Estimate FLOPs based on the layers in the ONNX model.
    This is a basic implementation and works for Conv and FC layers.
    More complex layers need specific handling.
    """
    onnx_model = onnx.load(onnx_model_path)
    flops = 0

    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            # Get the conv attributes and input/output shapes to estimate FLOPs
            conv_in_shape = [dim.dim_value for dim in node.input[0].type.tensor_type.shape.dim]
            conv_out_shape = [dim.dim_value for dim in node.output[0].type.tensor_type.shape.dim]
            kernel_shape = node.attribute[0].ints  # Assuming first attribute is kernel shape

            # FLOPs for Conv = 2 * kernel_size * output size * input channels * output channels
            kernel_flops = 2 * np.prod(kernel_shape)
            output_flops = np.prod(conv_out_shape)
            input_channels = conv_in_shape[1]  # C dimension (NCHW format)
            output_channels = conv_out_shape[1]  # C dimension (NCHW format)

            flops += kernel_flops * output_flops * input_channels * output_channels

        elif node.op_type == 'Gemm':
            # FLOPs for Fully connected (Gemm) = 2 * input_size * output_size
            fc_in_shape = [dim.dim_value for dim in node.input[0].type.tensor_type.shape.dim]
            fc_out_shape = [dim.dim_value for dim in node.output[0].type.tensor_type.shape.dim]

            flops += 2 * np.prod(fc_in_shape) * np.prod(fc_out_shape)
    flops = compute_magnitude_units(flops)
    return flops

def estimate_flops(onnx_model_path: str, inputs_shape=None) -> str:
    inputs = {}
    for input_shape in inputs_shape.items():
        inputs[input_shape[0]] = create_ndarray_f32(input_shape[1])
    #input_shape= {'src': create_ndarray_f32((1, 3, 1080, 1920))}
    # TODO: Implement dynamic input shape handling
    macs = model_profile(onnx_model_path, inputs, None)[0]
    flops =  macs*2 # FLOPs = 2*MACs
    flops = compute_magnitude_units(flops)
    return flops
    

def extract_onnx_model_stats(onnx_model_path: str, fixed_inputs_shape: dict={}) -> dict:
    """Extract ONNX model statistics: input/output dims, num params, size, and FLOPs."""
    inputs_dims, output_dims = get_input_output_dims(onnx_model_path)
    num_params = count_model_params(onnx_model_path)
    model_size = get_model_size(onnx_model_path)
    if fixed_inputs_shape != {}:
        flops = estimate_flops(onnx_model_path, inputs_shape=fixed_inputs_shape)
    else:
        try: 
            flops = estimate_flops(onnx_model_path, inputs_shape=inputs_dims)
        except:
            print("Error in estimating FLOPs")
            flops = 'N/A'
    return {
        "input_dims": inputs_dims,
        "output_dims": output_dims,
        "params": num_params,
        "model_size": model_size,
        "flops": flops if fixed_inputs_shape=={} else str(flops) + "@" + str({name:fixed_input_shape for name,fixed_input_shape in fixed_inputs_shape.items()}),
        "hash": generate_md5_hash(onnx_model_path)
    }


def edit_onnx_metadata(onnx_model_path, new_metadata, overwrite=False, output_path=None):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    
    # Get the current metadata
    existing_metadata = {meta.key: meta.value for meta in model.metadata_props}
    #print("Current Metadata:", existing_metadata)
    
    if overwrite:
        del model.metadata_props[:]  # Clear existing metadata
    
    for key, value in new_metadata.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value

    # Verify new metadata
    updated_metadata = {meta.key: meta.value for meta in model.metadata_props}
    #print("Updated Metadata:", updated_metadata)
    
    # Save the updated model
    save_path = output_path if output_path != None else onnx_model_path
    onnx.save(model, save_path)
    print(f"Updated ONNX model saved to {save_path}")



if __name__ == "__main__":

    # 1. Define all your models in a list of dictionaries
    # This makes it easy to add shapes or new models later without copying and pasting code
    models_to_test = [
        {
            "name": "3DDFA_v2",
            "path": 'models/face_alignment/3DDFA_v2_mb1_120x120.onnx',
            "fixed_shape": {}  # Add something like {'input.1': (1, 3, 120, 120)} if needed
        },
        {
            "name": "YuNet",
            "path": 'models/face_detection/face_detection_yunet_2023mar.onnx',
            "fixed_shape": {}
        },
        {
            "name": "scrfd10gkps",
            "path": 'models/face_detection/scrfd10gkps.onnx',
            "fixed_shape": {'input.1': (1, 3, 640, 640)}
        },
        {
            "name": "mobilenetv2arcface",
            "path": 'models/face_recognition/mobilenetv2arcface.onnx',
            "fixed_shape": {}
        },
        {
            "name": "GhostFaceNet_W1.3_S1_ArcFace",
            "path": 'models/face_recognition/GhostFaceNet_W1.3_S1_ArcFace.onnx',
            "fixed_shape": {}
        }
    ]

    # 2. Create an empty list to store all the results
    all_model_stats = []

    print("Starting batch extraction...\n")

    # 3. Loop through the models and extract stats
    for model_info in models_to_test:
        print(f"Processing {model_info['name']}...")
        
        # Call your extraction function
        stats = extract_onnx_model_stats(
            model_info['path'], 
            fixed_inputs_shape=model_info['fixed_shape']
        )
        
        # Add the model name into the dictionary so we know which is which
        stats['model_name'] = model_info['name']
        
        # Append the dictionary to our master list
        all_model_stats.append(stats)

    # 4. Print the final results cleanly
    print("\n" + "="*40)
    print("FINAL BATCH RESULTS")
    print("="*40)
    
    for stats in all_model_stats:
        print(f"\n--- {stats['model_name']} ---")
        for key, value in stats.items():
            if key != 'model_name': # Skip printing the name twice
                print(f"{key}: {value}")
