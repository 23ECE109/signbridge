"""
Export PyTorch Models to ONNX
==============================
Exports the trained encoder, decoder, and personalization MLP
to ONNX format for deployment with ONNX Runtime on Snapdragon.

Run after training:
    python training/export_onnx.py

Outputs (in models/):
    temporal_encoder.onnx
    translation_decoder.onnx
    personalization_mlp.onnx

Then quantize with:
    python training/quantize.py
"""

import logging
from pathlib import Path

log = logging.getLogger("signbridge.training.export")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

ROOT     = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR  = ROOT / "models"
OUT_DIR.mkdir(exist_ok=True)

INPUT_DIM   = 225
D_MODEL     = 256
SEQ_LEN     = 64
USER_DIM    = 64
VOCAB_SIZE  = 100   # Update to match actual vocabulary size after training


def export_encoder():
    try:
        import torch
        from training.train_encoder import build_encoder

        ckpt_path = CKPT_DIR / "temporal_encoder_best.pt"
        if not ckpt_path.exists():
            log.error(f"Checkpoint not found: {ckpt_path}")
            log.error("Run training first: python training/train_encoder.py")
            return False

        model = build_encoder()
        model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
        model.eval()

        dummy = torch.randn(1, SEQ_LEN, INPUT_DIM)
        out_path = str(OUT_DIR / "temporal_encoder.onnx")

        torch.onnx.export(
            model, dummy, out_path,
            input_names=["landmark_sequence"],
            output_names=["sequence_embedding"],
            dynamic_axes={"landmark_sequence": {0: "batch"}},
            opset_version=17,
        )
        log.info(f"✅  Encoder exported: {out_path}")
        return True

    except Exception as e:
        log.error(f"Encoder export failed: {e}")
        return False


def export_personalization_mlp():
    try:
        import torch
        from training.train_encoder import build_personalization_mlp

        ckpt_path = CKPT_DIR / "personalization_mlp_best.pt"
        if not ckpt_path.exists():
            log.warning(f"Personalization MLP checkpoint not found: {ckpt_path}")
            log.warning("Building fresh (untrained) MLP for export")

        model = build_personalization_mlp()
        if ckpt_path.exists():
            model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
        model.eval()

        dummy   = torch.randn(1, D_MODEL + USER_DIM)
        out_path = str(OUT_DIR / "personalization_mlp.onnx")

        torch.onnx.export(
            model, dummy, out_path,
            input_names=["fused_input"],
            output_names=["personalized_embedding"],
            dynamic_axes={"fused_input": {0: "batch"}},
            opset_version=17,
        )
        log.info(f"✅  Personalization MLP exported: {out_path}")
        return True

    except Exception as e:
        log.error(f"Personalization MLP export failed: {e}")
        return False


def export_decoder():
    try:
        import torch
        from training.train_encoder import build_decoder

        ckpt_path = CKPT_DIR / "translation_decoder_best.pt"
        if not ckpt_path.exists():
            log.warning(f"Decoder checkpoint not found: {ckpt_path}")
            return False

        model = build_decoder(VOCAB_SIZE)
        model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
        model.eval()

        dummy_memory = torch.randn(1, D_MODEL)
        dummy_tgt    = torch.randn(1, 1, D_MODEL)
        out_path     = str(OUT_DIR / "translation_decoder.onnx")

        torch.onnx.export(
            model, (dummy_memory, dummy_tgt), out_path,
            input_names=["context_embedding", "target_tokens"],
            output_names=["logits"],
            dynamic_axes={
                "context_embedding": {0: "batch"},
                "target_tokens":     {0: "batch", 1: "seq"},
                "logits":            {0: "batch", 1: "seq"},
            },
            opset_version=17,
        )
        log.info(f"✅  Decoder exported: {out_path}")
        return True

    except Exception as e:
        log.error(f"Decoder export failed: {e}")
        return False


def verify_exports():
    """Quick verification that exported models load and run."""
    try:
        import onnxruntime as ort
        import numpy as np

        for name, shape in [
            ("temporal_encoder.onnx",    (1, SEQ_LEN, INPUT_DIM)),
            ("personalization_mlp.onnx", (1, D_MODEL + USER_DIM)),
        ]:
            path = OUT_DIR / name
            if not path.exists():
                continue
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            inp  = {sess.get_inputs()[0].name: np.random.randn(*shape).astype(np.float32)}
            out  = sess.run(None, inp)
            log.info(f"✅  {name} — output shape: {out[0].shape}")

    except Exception as e:
        log.error(f"Verification failed: {e}")


if __name__ == "__main__":
    log.info("Exporting SignBridge models to ONNX...")
    export_encoder()
    export_personalization_mlp()
    export_decoder()
    log.info("\nVerifying exports...")
    verify_exports()
    log.info("\nDone. Next step: python training/quantize.py")
