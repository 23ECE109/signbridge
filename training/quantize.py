"""
ONNX Model Quantization for Snapdragon NPU
===========================================
Quantizes exported ONNX models to INT8 (encoder) or FP16 (decoder/ASR).
Output models are ready for QNNExecutionProvider on Hexagon HTP.

Strategy:
  temporal_encoder.onnx      → INT8  (high throughput, runs continuously)
  personalization_mlp.onnx   → FP16  (small, accuracy matters more)
  translation_decoder.onnx   → FP16  (autoregressive, keep precision)

Run after export:
    python training/quantize.py

Outputs:
    models/temporal_encoder_int8.onnx
    models/personalization_mlp_fp16.onnx
    models/translation_decoder_fp16.onnx
"""

import logging
from pathlib import Path

log = logging.getLogger("signbridge.training.quantize")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

ROOT    = Path(__file__).resolve().parent.parent
MDL_DIR = ROOT / "models"

INPUT_DIM  = 225
SEQ_LEN    = 64
D_MODEL    = 256
USER_DIM   = 64
N_CAL      = 100   # Number of calibration samples for INT8


def quantize_int8(model_path: Path, output_path: Path, input_shape: tuple) -> bool:
    """
    Static INT8 quantization using ONNX Runtime quantization tools.
    Requires calibration data to compute activation ranges.
    """
    try:
        import numpy as np
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

        class DummyCalibReader(CalibrationDataReader):
            """Generates synthetic calibration data — replace with real landmark data."""
            def __init__(self, input_name, shape, n=N_CAL):
                self.data = iter([
                    {input_name: np.random.randn(*shape).astype(np.float32)}
                    for _ in range(n)
                ])
            def get_next(self):
                return next(self.data, None)

        import onnxruntime as ort
        sess      = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inp_name  = sess.get_inputs()[0].name
        calib     = DummyCalibReader(inp_name, input_shape)

        quantize_static(
            model_input   = str(model_path),
            model_output  = str(output_path),
            calibration_data_reader = calib,
            quant_format  = None,          # QOperator format (compatible with QNN)
            weight_type   = QuantType.QInt8,
            activation_type = QuantType.QInt8,
        )
        log.info(f"✅  INT8 quantized: {output_path.name}")
        return True

    except ImportError:
        log.error("onnxruntime quantization tools not available.")
        log.error("Install: pip install onnxruntime-tools")
        return False
    except Exception as e:
        log.error(f"INT8 quantization failed for {model_path.name}: {e}")
        return False


def convert_fp16(model_path: Path, output_path: Path) -> bool:
    """
    Convert ONNX model weights to FP16 for Hexagon HTP FP16 mode.
    Activations remain FP32 at runtime; HTP handles the precision internally.
    """
    try:
        from onnxconverter_common import float16
        import onnx

        model    = onnx.load(str(model_path))
        fp16_mdl = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(fp16_mdl, str(output_path))
        log.info(f"✅  FP16 converted: {output_path.name}")
        return True

    except ImportError:
        log.warning("onnxconverter-common not installed — skipping FP16 conversion")
        log.warning("Install: pip install onnxconverter-common")
        # Fallback: copy original model (will run as FP32, still works)
        import shutil
        shutil.copy2(str(model_path), str(output_path))
        log.info(f"Copied original (FP32) as fallback: {output_path.name}")
        return True
    except Exception as e:
        log.error(f"FP16 conversion failed for {model_path.name}: {e}")
        return False


def verify_model(path: Path, input_shape: tuple, label: str) -> bool:
    """Run a quick inference check on the quantized model."""
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp  = {sess.get_inputs()[0].name: np.random.randn(*input_shape).astype(np.float32)}
        out  = sess.run(None, inp)
        log.info(f"  ✔  {label}: output shape {out[0].shape}, dtype {out[0].dtype}")
        return True
    except Exception as e:
        log.error(f"  ✗  {label} verification failed: {e}")
        return False


def main():
    log.info("=" * 55)
    log.info("  SignBridge — Model Quantization")
    log.info("=" * 55)

    results = {}

    # 1. Temporal encoder → INT8
    enc_in  = MDL_DIR / "temporal_encoder.onnx"
    enc_out = MDL_DIR / "temporal_encoder_int8.onnx"
    if enc_in.exists():
        results["encoder_int8"] = quantize_int8(enc_in, enc_out, (1, SEQ_LEN, INPUT_DIM))
    else:
        log.warning(f"Encoder not found: {enc_in} — run export_onnx.py first")
        results["encoder_int8"] = False

    # 2. Personalization MLP → FP16
    mlp_in  = MDL_DIR / "personalization_mlp.onnx"
    mlp_out = MDL_DIR / "personalization_mlp_fp16.onnx"
    if mlp_in.exists():
        results["mlp_fp16"] = convert_fp16(mlp_in, mlp_out)
    else:
        log.warning(f"Personalization MLP not found: {mlp_in}")
        results["mlp_fp16"] = False

    # 3. Translation decoder → FP16
    dec_in  = MDL_DIR / "translation_decoder.onnx"
    dec_out = MDL_DIR / "translation_decoder_fp16.onnx"
    if dec_in.exists():
        results["decoder_fp16"] = convert_fp16(dec_in, dec_out)
    else:
        log.warning(f"Decoder not found: {dec_in}")
        results["decoder_fp16"] = False

    # Verify
    log.info("\nVerifying quantized models...")
    if enc_out.exists():
        verify_model(enc_out, (1, SEQ_LEN, INPUT_DIM), "Encoder INT8")
    if mlp_out.exists():
        verify_model(mlp_out, (1, D_MODEL + USER_DIM),  "Personalization MLP FP16")

    log.info("\nQuantization summary:")
    for k, v in results.items():
        status = "✅ OK" if v else "❌ FAILED / SKIPPED"
        log.info(f"  {k:<25} {status}")

    log.info("\nDone. Deploy models from: models/")
    log.info("NPU deployment: use QNNExecutionProvider in ONNX Runtime")
    log.info("QNN config: htp_performance_mode=burst, enable_htp_fp16_precision=1")


if __name__ == "__main__":
    main()
