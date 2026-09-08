<div align="center">

# 🤝 SignBridge

### Private · Personalized · On-Device Communication Intelligence for Sign Language Users

[![Snapdragon Optimized](https://img.shields.io/badge/Snapdragon-X%20Elite-red?style=for-the-badge&logo=qualcomm&logoColor=white)](https://www.qualcomm.com/products/mobile/snapdragon/pcs-and-tablets/snapdragon-x-series)
[![Platform](https://img.shields.io/badge/Platform-Windows%2011%20ARM64-blue?style=for-the-badge&logo=windows&logoColor=white)](https://www.microsoft.com/windows)
[![Offline](https://img.shields.io/badge/Inference-100%25%20On--Device-green?style=for-the-badge&logo=cpu&logoColor=white)](#)
[![Python](https://img.shields.io/badge/Python-3.11%20ARM64-yellow?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)

**Snapdragon AI Lab Build & Present Challenge — Accessibility Track**

[🌐 Live Demo Page](https://23ECE109.github.io/signbridge) • [📄 Full Concept Document](docs/CONCEPT.md) • [🎥 Demo Video](#demo) • [📊 Slides](#presentation)

</div>

---

## What is SignBridge?

SignBridge is a **fully on-device, Snapdragon-powered communication system** for Deaf and hard-of-hearing individuals. It translates continuous sign language sequences into spoken speech in real time, and transcribes a hearing partner's speech into large text on screen — creating a **two-way communication bridge** that works offline, adapts to each user's signing style, and never transmits any data off the device.

> **One-line concept:**  
> SignBridge translates continuous sign language into speech and speech into accessible visual output, adapts to each user's signing style with a 5-minute enrollment, and actively handles ambiguity by asking for clarification — all on-device, all offline.

---

## The Problem

430 million people worldwide have disabling hearing loss. When a sign-language user interacts with someone who doesn't sign — at a **hospital, pharmacy, legal appointment, or emergency** — the communication gap is severe:

- 🔴 Human interpreters: expensive, unavailable on demand
- 🔴 Writing/typing: slow, dehumanizing, assumes English literacy
- 🔴 Existing AI apps: isolated gestures only, cloud-dependent, one-size-fits-all, one-way

**SignBridge is the first system that addresses all of these gaps together.**

---

## Why SignBridge is Different

| Feature | Google SL2T | Signapse | Other Research | **SignBridge** |
|---|---|---|---|---|
| Continuous signing | ✅ | ❌ | Partial | ✅ |
| **Uncertainty interaction** | ❌ | ❌ | ❌ | ✅ **3-tier system** |
| **Personal enrollment** | ❌ | ❌ | ❌ | ✅ **5-minute** |
| **Two-way bridge** | ❌ | ❌ | ❌ | ✅ |
| **Fully offline** | ❌ | ❌ | ❌ | ✅ |
| 100% on-device AI | ❌ (server translation) | ❌ (cloud API) | ❌ | ✅ |
| **Zero data transmission** | ❌ (landmarks sent) | ❌ | ❌ | ✅ |
| Correction memory | ❌ | ❌ | ❌ | ✅ |
| Windows / HP PC | ❌ (Android only) | ❌ (web) | ❌ | ✅ |

> **Novelty score: 7 / 10** — See [Full Analysis](docs/CONCEPT.md#section-24)

---

## Three Core Innovations

### 1. 🎯 Uncertainty-Aware Interaction
The AI does **not** pretend to understand every sign. It uses a three-tier confidence system:

```
Score ≥ 0.75  →  HIGH   →  Auto-translate and speak
Score 0.50–0.74 →  MED  →  Ask: "Did you mean A or B?"
Score < 0.50   →  LOW   →  Request the sign again
```

No deployed sign-language system exposes uncertainty as a communication interaction loop.

### 2. 👤 5-Minute Personal Enrollment
Sign once, get recognized like yourself.

- Sign 30 reference signs during a guided setup (5 minutes)
- System derives a **64-dim user embedding** specific to your signing style
- Personalization layer adapts the model to your speed, hand size, regional dialect, and motor patterns
- Cross-session **correction memory** improves recognition over time

### 3. 🔄 Two-Way Communication Bridge
```
Deaf user signs  →  [SignBridge]  →  Speech output (hearing partner hears)
Hearing partner speaks  →  [SignBridge]  →  Large text (Deaf user reads)
```
Both directions. Offline. No network. No cloud.

---

## System Architecture

```
┌─────────────────── SNAPDRAGON X ELITE / HP PC ─────────────────────┐
│                                                                       │
│  Camera  ──▶  MediaPipe Holistic (NPU · FP16)                        │
│               543 landmarks/frame @ 30 FPS · 52–76 µs/frame          │
│                    │                                                   │
│                    ▼                                                   │
│           Temporal Encoder (NPU/CPU · INT8)                           │
│           4-layer Transformer · d=256 · 64-frame window               │
│                    │                                                   │
│         ┌──────────┴──────────┐                                       │
│    User Profile DB       Context Window                                │
│    (AES-256 · local)     (last 5 turns)                                │
│         └──────────┬──────────┘                                       │
│                    │                                                   │
│           Personalization Layer (CPU)                                  │
│           user_embedding [64] + seq_embedding [256] → MLP             │
│                    │                                                   │
│           Translation Decoder (CPU/NPU)                               │
│           + Confidence Estimator (3-tier)                              │
│                    │                                                   │
│        ┌───────────┼───────────┐                                      │
│        ▼           ▼           ▼                                      │
│     TTS Output  Disambig    Re-sign                                    │
│   (piper-tts)   Dialog      Prompt                                    │
│        │           │                                                   │
│        └───────────▶  Correction Memory (SQLite · kNN)                │
│                                                                       │
│  Microphone ──▶  Distil-Whisper (NPU · FP16)  ──▶  Text Display     │
└───────────────────────────────────────────────────────────────────────┘
```

**NPU handles**: landmark extraction + ASR (proven on Qualcomm AI Hub)  
**CPU handles**: personalization MLP, context engine, correction memory  
**Nothing leaves the device**

---

## AI Models

| Role | Model | Format | Source |
|------|-------|--------|--------|
| Hand/body/face landmarks | MediaPipe Holistic | ONNX / QNN | [Qualcomm AI Hub](https://aihub.qualcomm.com/models/mediapipe_hand_gesture) |
| ASR (hearing partner) | Distil-Whisper Small | ONNX FP16 | [Qualcomm AI Hub](https://aihub.qualcomm.com/models/distil_whisper) |
| Temporal encoder | Custom Transformer 4L | ONNX INT8 | Trained on WLASL + custom |
| Translation decoder | Custom Transformer 2L | ONNX INT8 | Trained on WLASL + custom |
| Personalization MLP | Custom 2-layer MLP | ONNX | Trained on enrollment data |
| TTS (speech output) | piper-tts en_US-lessac | ONNX | [rhasspy/piper](https://github.com/rhasspy/piper) |
| Summarization (optional) | Phi-3.5-Mini 4-bit | GenieX/GGUF | Qualcomm GenieX |

---

## Hardware Requirements

- **Target**: HP OmniBook X 14 / HP EliteBook Ultra (Snapdragon X Elite)
- **Minimum**: Any Snapdragon X / Copilot+ PC with Hexagon NPU
- Built-in RGB camera (720p+)
- Built-in microphone
- 16 GB RAM (8 GB minimum, core pipeline only)
- Windows 11 ARM64 (Copilot+)
- No internet required for operation

---

## Getting Started

### Prerequisites

```powershell
# 1. Install miniforge (ARM64 Python for Windows)
# https://github.com/conda-forge/miniforge/releases

# 2. Create environment
conda create -n signbridge python=3.11 -y
conda activate signbridge

# 3. Install dependencies
pip install -r requirements.txt
```

### Run SignBridge

```powershell
# Activate environment
conda activate signbridge

# Launch the application
python src/main.py
```

### First-Time Enrollment

On first launch, SignBridge will guide you through a **5-minute enrollment**:
1. Follow the on-screen sign prompts (30 reference signs)
2. Your user profile is saved locally and encrypted
3. SignBridge is now calibrated to your signing style

---

## Project Structure

```
signbridge/
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
├── LICENSE
│
├── docs/
│   ├── CONCEPT.md               ← Full 36-section concept document
│   ├── ARCHITECTURE.md          ← Detailed system architecture
│   └── DATASET.md               ← Dataset strategy and collection protocol
│
├── src/
│   ├── main.py                  ← Application entry point
│   │
│   ├── pipeline/
│   │   ├── camera.py            ← Camera capture & frame buffer
│   │   ├── landmark_extractor.py ← MediaPipe Holistic (NPU)
│   │   └── orchestrator.py      ← Pipeline coordination
│   │
│   ├── models/
│   │   ├── temporal_encoder.py  ← Transformer encoder (ONNX)
│   │   ├── translation_decoder.py ← Decoder + token generation
│   │   ├── confidence.py        ← 3-tier confidence estimator
│   │   └── tts.py               ← piper-tts wrapper
│   │
│   ├── personalization/
│   │   ├── enrollment.py        ← 5-minute enrollment flow
│   │   ├── user_embedding.py    ← User profile management
│   │   └── correction_memory.py ← kNN correction store (SQLite)
│   │
│   ├── context_engine/
│   │   ├── context_window.py    ← Sliding conversation history
│   │   └── topic_tag.py         ← Session topic bias
│   │
│   └── ui/
│       ├── app.py               ← Main application window (PyQt6)
│       ├── disambiguation.py    ← Disambiguation dialog
│       ├── enrollment_ui.py     ← Enrollment flow UI
│       └── two_way_display.py   ← Partner speech text display
│
├── training/
│   ├── train_encoder.py         ← Temporal encoder training script
│   ├── train_decoder.py         ← Translation decoder training
│   ├── train_personalization.py ← Personalization MLP training
│   ├── export_onnx.py           ← PyTorch → ONNX export
│   └── quantize.py              ← ONNX INT8/FP16 quantization
│
├── data/
│   ├── vocabulary.json          ← 150-sign vocabulary definition
│   ├── preprocessing.py         ← Landmark normalization utilities
│   └── augmentation.py          ← Data augmentation (jitter, speed, flip)
│
└── web/
    └── index.html               ← Project landing page (GitHub Pages)
```

---

## Dataset

SignBridge uses a custom **150-sign focused vocabulary** designed for high-priority communication scenarios:

- 🚨 Emergency (15 signs)
- 🏥 Medical (25 signs)
- 🗣️ Daily interaction (40 signs)
- ❓ Questions & responses (20 signs)
- 🔤 Grammar connectors (30 signs)
- 🔤 Fingerspelling (26 letters)

**Collection protocol**: 8 participants × 5 repetitions × 3 backgrounds × 2 lighting × 2 handedness = ~10,000 clips

Pre-training uses [WLASL](https://github.com/dxli94/WLASL) (2,000 signs, CC-BY) and [PHOENIX-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) (research license).

See [docs/DATASET.md](docs/DATASET.md) for full protocol.

---

## Privacy

> **Your conversation never leaves this device.**

| Principle | Implementation |
|-----------|----------------|
| No raw video stored | Circular RAM buffer only, never written to disk |
| No data transmission | Zero network calls in the inference pipeline |
| Encrypted user profile | AES-256, key via Windows DPAPI |
| Encrypted correction memory | Same SQLite DB, same encryption |
| Conversation history | RAM only, cleared on app close |
| One-click data deletion | Settings → Delete My Profile |

---

## Evaluation

| Metric | Target |
|--------|--------|
| Sign recognition (top-1, 150 signs) | > 80% |
| End-to-end latency (sign → speech) | < 1.5 seconds |
| Landmark extraction FPS | ≥ 25 FPS |
| RAM usage (core pipeline) | < 600 MB |
| ASR accuracy (quiet speech) | > 90% WER |
| Personalization improvement | ≥ +10% for enrolled user |
| Offline functionality | 100% |

---

## Why Snapdragon?

| | Cloud Approach | SignBridge (Snapdragon Local) |
|---|---|---|
| Latency | 200–800 ms (network + server) | < 100 ms (local NPU) |
| Privacy | Coordinates/data sent to server | Zero data leaves device |
| Offline | ❌ Not possible | ✅ Fully functional |
| Medical/legal use | Requires HIPAA compliance, BAAs | No cloud = no compliance barrier |
| Power | Network radio active continuously | NPU fraction of CPU power draw |

The Hexagon NPU delivers MediaPipe landmark extraction at **52–76 µs per frame** — roughly 1000× within the 33 ms/frame budget. This headroom is what makes real-time, low-latency translation possible.

---

## Demo

> 🎥 **[Watch Demo Video](#)** *(link added after recording)*

**2-minute demo scenario**: A Deaf user at a pharmacy  
1. Signs a continuous sentence → speech output
2. Ambiguous sign → disambiguation dialog appears
3. Pharmacist speaks → live transcript for Deaf user
4. User corrects a mistake → system remembers for next time
5. Wi-Fi turned off live → everything still works

---

## Presentation

> 📊 **[View Slides](#)** *(link added after upload)*

---

## Development Roadmap

| Phase | Timeline | Status |
|-------|----------|--------|
| Foundation (NPU pipeline setup) | Weeks 1–2 | 🔧 In Progress |
| Custom dataset collection | Weeks 2–4 | 📋 Planned |
| Model training + export | Weeks 3–5 | 📋 Planned |
| Full integration | Weeks 5–7 | 📋 Planned |
| Evaluation + optimization | Weeks 7–8 | 📋 Planned |
| Demo polish + presentation | Weeks 8–10 | 📋 Planned |

---

## Team

| Role | Responsibilities |
|------|-----------------|
| AI / ML Engineer | Dataset prep, Transformer training, ONNX export, NPU quantization, confidence estimator |
| Systems / App Engineer | Pipeline integration, UI (PyQt6), Whisper/TTS, SQLite storage, demo |
| Data & Evaluation *(3-person team)* | Dataset recording, quality control, metrics, slides |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

Dataset components are subject to their respective licenses (WLASL: CC-BY, PHOENIX: research only).

---

## Citation / Acknowledgements

- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — MediaPipe models, Distil-Whisper, Snapdragon toolchain
- [WLASL Dataset](https://github.com/dxli94/WLASL) — Li et al., CC-BY license
- [PHOENIX-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) — RWTH Aachen
- [piper-tts](https://github.com/rhasspy/piper) — rhasspy, MIT license
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — Google, Apache 2.0
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — Microsoft, MIT license

---

<div align="center">

**Built for the Snapdragon AI Lab Build & Present Challenge**  
*Accessibility · Privacy · On-Device AI*

</div>
