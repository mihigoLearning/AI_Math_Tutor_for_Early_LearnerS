"""Build a tiny quantized ONNX language-head model for offline packaging.

The assignment accepts `model.onnx` OR `.gguf`. We generate a real ONNX model
with int8-quantized weights (stored as int8 + scale/zero-point and dequantized
in-graph), then save it to `tutor/model.onnx`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _qlinear_weight(name: str, arr: np.ndarray, scale: float):
    """Return (q_weight, scale, zero_point, dequant_node, dequant_out_name)."""
    zp = np.int8(0)
    q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)

    w_name = f"{name}_q"
    s_name = f"{name}_scale"
    z_name = f"{name}_zero"
    out_name = f"{name}_deq"

    w_init = numpy_helper.from_array(q, name=w_name)
    s_init = numpy_helper.from_array(np.array([scale], dtype=np.float32), name=s_name)
    z_init = numpy_helper.from_array(np.array([zp], dtype=np.int8), name=z_name)

    dq = helper.make_node(
        "DequantizeLinear",
        inputs=[w_name, s_name, z_name],
        outputs=[out_name],
        name=f"Dequantize_{name}",
    )
    return (w_init, s_init, z_init, dq, out_name)


def build_model(out_path: Path) -> None:
    rng = np.random.default_rng(20260424)

    in_dim = 16
    hidden = 12
    out_dim = 5  # counting, number_sense, addition, subtraction, word_problem

    # Tiny MLP language head: X -> Relu(XW1 + b1) -> logits(XW2 + b2)
    w1 = (rng.normal(0, 0.2, size=(in_dim, hidden))).astype(np.float32)
    b1 = (rng.normal(0, 0.1, size=(hidden,))).astype(np.float32)
    w2 = (rng.normal(0, 0.2, size=(hidden, out_dim))).astype(np.float32)
    b2 = (rng.normal(0, 0.1, size=(out_dim,))).astype(np.float32)

    w1_q, w1_s, w1_z, dq1, w1_deq = _qlinear_weight("w1", w1, scale=0.02)
    w2_q, w2_s, w2_z, dq2, w2_deq = _qlinear_weight("w2", w2, scale=0.02)

    b1_init = numpy_helper.from_array(b1, name="b1")
    b2_init = numpy_helper.from_array(b2, name="b2")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, in_dim])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, out_dim])

    nodes = [
        dq1,
        dq2,
        helper.make_node("MatMul", ["input", w1_deq], ["h1_mm"], name="MatMul1"),
        helper.make_node("Add", ["h1_mm", "b1"], ["h1"], name="Add1"),
        helper.make_node("Relu", ["h1"], ["h1_relu"], name="Relu1"),
        helper.make_node("MatMul", ["h1_relu", w2_deq], ["out_mm"], name="MatMul2"),
        helper.make_node("Add", ["out_mm", "b2"], ["logits"], name="Add2"),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="NumeracyLanguageHeadInt8",
        inputs=[x],
        outputs=[y],
        initializer=[w1_q, w1_s, w1_z, w2_q, w2_s, w2_z, b1_init, b2_init],
    )
    model = helper.make_model(
        graph,
        producer_name="aims-math-tutor",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path)


if __name__ == "__main__":  # pragma: no cover
    root = Path(__file__).resolve().parents[1]
    out = root / "tutor" / "model.onnx"
    build_model(out)
    print(f"Wrote quantized model: {out}")
