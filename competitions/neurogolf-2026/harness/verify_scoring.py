"""Empirically execute the REAL neurogolf_utils scoring path to verify my claims.

We avoid verify_network()'s IPython/onnx_tool display branch and instead call the
same authoritative pieces it uses: single_layer_conv2d_network, check_network,
sanitize_model, verify_subset, score_network, and the points formula.
"""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ng_data"))
import numpy as np
import onnx, onnxruntime
import neurogolf_utils as ng

print("== constants from utils ==")
print("GRID_SHAPE      :", ng._GRID_SHAPE)            # expect [1,10,30,30]
print("EXCLUDED_OPS    :", ng._EXCLUDED_OP_TYPES)
print("FILESIZE_LIMIT  :", ng._FILESIZE_LIMIT_IN_BYTES, "bytes (=1.44*1024*1024)")
print("OPSET / IR      :", ng._IR_VERSION, ng._OPSET_IMPORTS)

def identity_w(o, i, off):     # 1.0 on the diagonal, center tap only
    return 1.0 if (o == i and off == (0, 0)) else 0.0

def score_one(net, examples, tag, task_num=900):
    fn = f"task{task_num:03d}.onnx"
    onnx.save(net, fn)
    assert ng.check_network(fn), "filesize check failed"
    sanitized = ng.sanitize_model(onnx.load(fn))
    opts = onnxruntime.SessionOptions()
    opts.enable_profiling = True
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.profile_file_prefix = f"{task_num:03}"
    sess = onnxruntime.InferenceSession(sanitized.SerializeToString(), opts)
    r, w, _ = ng.verify_subset(sess, examples["train"] + examples["test"])
    rg, wg, _ = ng.verify_subset(sess, examples["arc-gen"])
    memory, params = ng.score_network(sanitized, sess.end_profiling())
    points = max(1.0, 25.0 - math.log(max(1.0, memory + params)))
    print(f"\n== {tag} ==")
    print(f"correctness: ARC-AGI {r} pass/{w} fail | ARC-GEN {rg} pass/{wg} fail")
    print(f"params={params}  memory={memory} bytes  cost=memory+params={memory+params}")
    print(f"points = max(1, 25 - ln(cost)) = {points:.4f}")
    return params, memory, points

# An identity task: output == input, over a few small grids (one-hot padded to 30x30).
def grid(colors):
    return colors
ident_examples = {
    "train": [{"input": g, "output": g} for g in [
        [[1,2],[3,4]], [[5,0,6],[7,8,9]], [[2,2],[2,2]]]],
    "test":  [{"input": [[9,8],[7,6]], "output": [[9,8],[7,6]]}],
    "arc-gen": [{"input": [[3,3,3],[3,1,3],[3,3,3]], "output": [[3,3,3],[3,1,3],[3,3,3]]}],
}

# k=1 identity conv -> expect 100 params (10*10*1*1), perfect identity, ~0 intermediate memory.
net1 = ng.single_layer_conv2d_network(identity_w, kernel_size=1)
score_one(net1, ident_examples, "identity 1x1 conv", 900)
print("calculate_params(net1) =", ng.calculate_params(net1), "(expect 100)")

# k=3 identity conv -> expect 900 params (10*10*3*3); still identity (only center tap=1).
net3 = ng.single_layer_conv2d_network(identity_w, kernel_size=3)
score_one(net3, ident_examples, "identity 3x3 conv", 901)
print("calculate_params(net3) =", ng.calculate_params(net3), "(expect 900)")

# Confirm a banned op is rejected by score_network.
import onnx.helper as oh
x = oh.make_tensor_value_info("input", ng._DATA_TYPE, ng._GRID_SHAPE)
y = oh.make_tensor_value_info("output", ng._DATA_TYPE, ng._GRID_SHAPE)
nz = oh.make_node("NonZero", ["input"], ["output"])
bad = oh.make_model(oh.make_graph([nz], "g", [x], [y]),
                    ir_version=ng._IR_VERSION, opset_imports=ng._OPSET_IMPORTS)
m, p = ng.score_network(bad, "/dev/null")
print("\n== banned-op check ==")
print(f"score_network on NonZero graph -> (memory={m}, params={p}) ; expect (None, None)")
