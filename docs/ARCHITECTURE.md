# SignBridge — Detailed Architecture Reference

## Compute Unit Breakdown

| Component | Runs On | Format | Latency Target |
|-----------|---------|--------|----------------|
| MediaPipe Hand Detection | Hexagon NPU (HTP FP16) | ONNX via QNN | 52–76 µs/frame |
| MediaPipe Pose Estimation | Hexagon NPU (HTP FP16) | ONNX via QNN | ~100 µs/frame |
| Frame buffer (64 frames) | RAM only, never disk | — | — |
| Temporal Encoder (4L, d=256) | NPU + CPU split | ONNX INT8 | ~5–15 ms/window |
| Personalization MLP (2L) | CPU | ONNX | < 1 ms |
| Context Engine (TF-IDF) | CPU | NumPy | < 1 ms |
| Translation Decoder (2L) | CPU (+ ONNX) | ONNX | ~5–20 ms |
| Confidence Estimator | CPU | NumPy | < 1 ms |
| piper-tts | CPU (ARM64 binary) | ONNX | ~200–400 ms |
| Distil-Whisper (ASR) | Hexagon NPU (HTP FP16) | ONNX via QNN | ~100–300 ms/chunk |
| Correction Memory (kNN) | CPU | SQLite + NumPy | < 5 ms |

**Total core model weights**: ~330 MB  
**Target RAM at runtime**: < 600 MB  
**Total pipeline latency (sign-end → speech-start)**: < 1.5 seconds

---

## ONNX Runtime QNN Configuration

```python
import onnxruntime as ort

providers = [
    (
        "QNNExecutionProvider",
        {
            "backend_type":                             "htp",
            "htp_performance_mode":                     "burst",
            "htp_graph_finalization_optimization_mode": "3",
            "enable_htp_fp16_precision":                "1",
            "offload_graph_io_quantization":            "1",
        }
    ),
    "CPUExecutionProvider",
]

session = ort.InferenceSession("model.onnx", providers=providers)
```

---

## Model Conversion Pipeline

```
1. Train in PyTorch (GPU, training machine)
   └─ python training/train_encoder.py
   └─ python training/train_decoder.py

2. Export to ONNX
   └─ python training/export_onnx.py

3. Quantize (INT8 for encoder, FP16 for decoder)
   └─ python training/quantize.py
   # Uses ONNX quantization tools or Qualcomm AIMET

4. Compile for QNN (optional — for maximum NPU utilization)
   └─ qnn-onnx-converter --input_network model.onnx --output_path model.dlc

5. Validate on Snapdragon device
   └─ python -c "import onnxruntime; ..."
```

---

## Data Flow (signed message lifecycle)

```
t=0ms    Camera frame arrives (640×480 RGB)
t=0.1ms  MediaPipe: detect hands + pose → 225 landmarks extracted (NPU: 52-76µs)
t=33ms   Next frame: buffer = 64 frames = 2.1 seconds of signing
t=40ms   Normalize landmark sequence: zero-mean, scale-invariant, pad to [64, 225]
t=45ms   Temporal Encoder: [64, 225] → [256] embedding (NPU: ~10ms)
t=47ms   Personalization MLP: [256]+[64] → [256] (CPU: <1ms)
t=48ms   Context Engine: append [32] context vector (CPU: <1ms)
t=70ms   Translation Decoder: [256+32] → top-3 [(text, prob)] (CPU: ~20ms)
t=71ms   Confidence score: 0.82 → HIGH tier (CPU: <1ms)
t=72ms   Correction memory lookup: no override (CPU: <5ms)
t=72ms   Final text: "Do I need to take the medicine in the morning?"
t=300ms  piper-tts: synthesis + playback starts (CPU: ~200ms synthesis)
         → Total sign-to-speech latency: ~300ms (synthesis) or ~72ms (text-to-screen)
```

---

## Privacy Architecture

```
Camera frame  (RAM circular buffer, 64 frames max)
     │ ← NEVER written to disk
     ▼
Landmark coordinates  (RAM only during inference)
     │ ← NEVER transmitted
     ▼
Sequence embedding  (RAM only during session)
     │
     ▼
Translation text  (displayed on screen, spoken via TTS)
     │
     ▼ (on user correction only)
Correction memory  (AES-256 encrypted SQLite, local disk)
     │
     ▼ (on enrollment only)
User profile  (AES-256 encrypted JSON, local disk)

Network calls during normal operation: ZERO
```

---

## Snapdragon X Elite SoC Specifics

- **NPU**: Hexagon HTP (Hexagon Tensor Processor)
- **Capability**: 45 TOPS (INT8), >20 TOPS (FP16)
- **Memory bandwidth**: ~136 GB/s (LPDDR5x)
- **Relevant QNN backend**: `kHtpBackend`, `kHtpFp16`
- **ONNX Runtime integration**: `QNNExecutionProvider` (available in onnxruntime >= 1.16)
- **Windows integration**: Works with Windows 11 ARM64, no additional driver installation required on Copilot+ PCs

---

## Training Architecture (PyTorch reference)

```python
import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    """4-layer Transformer encoder for landmark sequences."""
    def __init__(self, input_dim=225, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed  = nn.Embedding(64, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool        = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: [B, T, input_dim]
        x  = self.input_proj(x)                          # [B, T, d_model]
        t  = torch.arange(x.size(1), device=x.device)
        x  = x + self.pos_embed(t).unsqueeze(0)          # Add positional embeddings
        x  = self.transformer(x)                          # [B, T, d_model]
        x  = self.pool(x.transpose(1, 2)).squeeze(-1)    # [B, d_model]
        return x


class PersonalizationMLP(nn.Module):
    """2-layer MLP: [320] → [256]. Fuses sequence + user embeddings."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256 + 64, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
        )

    def forward(self, seq_emb, user_emb):
        combined = torch.cat([seq_emb, user_emb], dim=-1)
        return self.net(combined)
```
