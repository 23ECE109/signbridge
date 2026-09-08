# SignBridge — Full Concept Document
### Snapdragon AI Lab Build & Present Challenge
#### Version 1.0 | September 2026

---

> **Brutally honest prior-art analysis → differentiated architecture → realistic MVP**  
> This document covers all 36 required sections.

---

## PRIOR-ART & COMPETITOR RESEARCH

### What was found (and why it matters before finalising the concept)

Before writing a single line of architecture, a serious search was performed across commercial products, research papers, hackathon projects, and platform-specific models. The findings materially shaped the final concept.

**Google DeepMind SL2T (August 2026 — Pixel 11)**  
The most recent and most relevant competitor. SL2T tracks 130 upper-body landmarks on-device and sends those coordinates (not raw video) to a server for translation into English text. It powers Gboard and Live Transcribe on Pixel 11. Trained on over 100,000 hours of signing data across 50+ sign languages, launching with ASL → English. It is gloss-free (no intermediate label bottleneck), which is a genuine technical advance. However, the translation computation runs server-side, it is Android/Pixel-only, it supports no user personalization, it offers no two-way communication bridge, it provides no confidence/uncertainty signal to the user, there is no user feedback or correction loop, and there is no offline operation for the core translation. This is the benchmark to beat.

**Signapse (UK/US, commercial)**  
Text-to-sign avatar generation for broadcast and web content (English → BSL/ASL digital signer). This is the reverse direction (text → sign production) for accessibility content, not for real-time user communication. Not a personal communication tool.

**SignAll (startup)**  
Used multiple sensors (depth cameras, glove sensors in early versions) for workplace sign-language communication. Required specialized hardware, was expensive, not widely deployed, and had no publicly documented personalization or uncertainty features.

**Spotter+GPT (research, 2024)**  
Two-stage pipeline: a sign spotter identifies individual signs, then an LLM converts the spotted glosses into natural sentences. Clever but still gloss-dependent in the first stage, requires cloud LLM, no personalization, no two-way communication, no offline.

**Toward Real-Time Sentence-Level SLT (arxiv 2607.09611, 2026)**  
SHuBERT+ByT5 stack fine-tuned on How2Sign, runs perception and translation on a CPU/GPU backend, with a Raspberry Pi client for capture. BLEU 15.9 on test set. Key finding: mean post-finalization latency 1.354s, P95 2.130s. Server-dependent, no personalization, ASL only, no offline, no two-way.

**Bidirectional Sign Language Translator (IJERT, 2026)**  
Recognizes gestures and converts typed text into sign animations. Two-way in concept but implemented as isolated gesture → text + text → animation. No continuous signing, no context awareness, no personalization.

**Few-Shot Prototypical Networks for ASL (arxiv 2512.10562)**  
Few-shot classification of isolated ASL signs using metric learning. Isolated signs only, classification not translation, no context, no sequence understanding.

**UPRet — Uncertainty-aware SLT (arxiv 2405.19689)**  
Models sign-language-to-text retrieval as probability distributions to capture inherent ambiguity. This is uncertainty modeling at the retrieval level, not at the real-time communication interaction level. Research paper, not a deployed system.

**Adaptive Simultaneous SLT with Confident Translation Length (ACL 2024)**  
Introduces "confident translation length" for streaming sign video to decide when to output vs. wait. Important streaming contribution, but at the model output level, not exposed to the user as an interaction design feature.

**Mobile AI Application for Continuous SLR (Springer 2026)**  
Transformer-based on-device model for pre-recorded and real-time signing on mobile. Second model trained on SIGNUM Database isolated signs. No personalization, no context engine, no uncertainty UI, no two-way.

**Qualcomm AI Hub — MediaPipe Hand Gesture Recognition**  
Available on Qualcomm AI Hub: MediaPipe hand landmark detector optimized for Hexagon HTP backend, FP16 precision, ~52–76 µs inference time per frame. This is available for Windows ARM/Snapdragon. Also available: MediaPipe Pose Estimation, Distil-Whisper (ASR), Whisper-Base, and GenieX (on-device LLM inference for Snapdragon).

**What is confirmed NOT to exist as a deployed, combined system:**  
No current product or research system combines all of: (a) continuous sign-to-speech on a Windows/PC platform, (b) on-device full inference (no server required), (c) user-personalized signing style adaptation with a small enrollment set, (d) interactive uncertainty dialogue (the system asks for clarification when unsure), (e) a two-way communication bridge with live speech-to-text and visual display for the non-signing partner, and (f) user correction feedback that persists across sessions — all running offline on Snapdragon-powered hardware.

---

## SECTION 1 — Final Project Name

**SignBridge**  
*Subtitle: Private, Personalized, On-Device Communication Intelligence for Sign Language Users*

The name was changed from "SignSpeak" because "speak" implies one-directional output. "Bridge" explicitly signals two-way communication and the bridging of a real social gap.

---

## SECTION 2 — One-Line Concept

> SignBridge is a fully on-device, Snapdragon-powered communication system that translates continuous sign language into speech and speech into accessible visual output, adapts to each user's signing style with a 5-minute enrollment, and actively handles ambiguity by asking for clarification — all without requiring an internet connection.

---

## SECTION 3 — Real-World Problem

Approximately 430 million people worldwide have disabling hearing loss (WHO, 2021). A significant portion of Deaf and hard-of-hearing individuals use sign language as their primary language. When these individuals communicate with hearing people who do not know sign language — at a hospital, pharmacy, government office, job interview, or emergency room — the communication gap is severe.

Existing solutions fall into three categories, all inadequate:

1. **Human interpreters** — expensive, unavailable on demand, unavailable in emergencies, and require advance scheduling. Interpreter shortages are well-documented in every country.

2. **Typing/writing** — slow, dehumanizing for conversational exchange, and impossible for people with limited written language literacy (because signed languages have different grammar from written English).

3. **Current sign-language apps** — either recognize isolated gestures one at a time (not natural signing), or require cloud connectivity (unusable in poor-coverage areas, privacy risk when the conversation is medical or legal), or are locked to specific devices (Google SL2T is Pixel 11 only as of August 2026), or provide no adaptation to the individual's signing style (a signer with motor differences, or who uses a regional dialect, may be poorly recognized), or offer no way to complete a two-way conversation.

The result: a sign-language user in a real-world encounter with a non-signer is frequently left without any usable communication tool.

---

## SECTION 4 — Why Existing Solutions Are Insufficient

| Gap | Evidence |
|-----|----------|
| Sign-to-text only, not sign-to-speech | SL2T, all academic systems — output is text, which a hearing person still has to read. In a face-to-face conversation this is awkward. |
| No two-way bridge | None of the reviewed systems handle the non-signer's speech response and convert it back to accessible output for the Deaf user. |
| Cloud-dependent translation | SL2T sends landmarks to a server. Network failure = no service. |
| No personalization | SL2T, all academic systems — the model is one-size-fits-all. A signer with Parkinson's, motor differences, or a regional dialect will be systematically under-served. |
| No uncertainty interaction | No deployed system asks "Did you mean X or Y?" when a sign is ambiguous. The system either outputs wrong text silently or throws an error. |
| No user feedback loop | No deployed system learns from correction. Every session starts from scratch. |
| PC/Windows gap | All solutions target mobile (Android/iOS) or specialized hardware. No solution targets HP Snapdragon PCs, which are used in enterprise, healthcare, and education settings where accessibility matters most. |
| Privacy | Even SL2T sends coordinates to a server. A medical conversation's content — reconstructable from dense landmark sequences — should not leave the device. |

---

## SECTION 5 — Exact Innovation / Gap

SignBridge's differentiator is not any single feature — it is the **first integrated communication system** that combines:

1. **Continuous sign sequence → spoken speech** (not just text) on a Windows Snapdragon PC, fully offline
2. **Two-way bridge**: incoming speech → text + visual display for the Deaf user, closing the loop
3. **5-minute personal enrollment** that adapts the recognition model to an individual's signing style using lightweight user embeddings — without retraining
4. **Uncertainty-aware interaction**: three confidence tiers — auto-translate, ask for disambiguation, request re-sign — with natural language prompts
5. **Persistent user correction memory**: corrections stored as user-specific embedding adjustments and retrieved in future sessions
6. **Snapdragon NPU-accelerated pipeline** with all AI running on the local Hexagon NPU + CPU, zero server dependency

The combination of (3) + (4) + (5) across a full two-way offline pipeline is what does not exist anywhere.

---

## SECTION 6 — Proposed Solution

SignBridge is a Windows desktop application that runs on any Snapdragon X / Copilot+ HP PC. It uses the device's built-in camera and microphone. No additional hardware is required.

**For the signing user:**
The camera continuously observes the user. A local pose/hand/face landmark extraction model (MediaPipe Holistic, quantized for Snapdragon HTP) extracts 543 keypoints per frame at real-time framerate. These landmark sequences feed into a temporal understanding model (lightweight Transformer encoder) that encodes gesture sequences into embeddings. A personalization layer fuses the user's enrolled signing style embedding with the sequence embedding. A context engine (sliding window of recent utterances + conversation topic) refines interpretation. A confidence estimator outputs a three-tier decision. High confidence → TTS speaks the translation aloud. Medium confidence → system asks a disambiguation question in natural language. Low confidence → system displays an animated prompt asking the user to re-sign. User corrections are stored as embedding adjustments and recalled in future sessions.

**For the hearing/non-signing conversation partner:**
The microphone captures their speech. Distil-Whisper (quantized for Snapdragon HTP) transcribes it in real time. The transcript is displayed in large, clear text on screen for the Deaf user. Optionally, a lightweight on-device TTS model can produce a visual representation or rephrase the speech.

**The entire pipeline runs locally. Nothing leaves the device.**

---

## SECTION 7 — Detailed Working Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      SIGNING USER SIDE                       │
└─────────────────────────────────────────────────────────────┘

[1] Camera captures video at 30 FPS
    ↓
[2] Frame Buffer (sliding 64-frame window, ~2.1 seconds)
    Raw frames are never written to disk.
    ↓
[3] MediaPipe Holistic Landmark Extractor
    → 21 hand landmarks × 2 hands × 3D = 126 coordinates/frame
    → 33 body pose landmarks × 3D = 99 coordinates/frame
    → 468 face mesh landmarks (optional, for non-manual markers)
    → Output: landmark sequence tensor  [T × 225] per window
    ↓
[4] Temporal Encoder (Transformer, 4-layer, d_model=256)
    → Trained on: WLASL + PHOENIX-2014T + custom collected data
    → Input: landmark sequence
    → Output: sequence embedding [256-dim]
    ↓
[5] Personalization Fusion Layer
    → User profile loaded: user_embedding [64-dim]
    → Concatenate [sequence_embedding || user_embedding] → 320-dim
    → Lightweight MLP → refined_embedding [256-dim]
    ↓
[6] Context Engine
    → Last 5 utterances stored as text in a local sliding context window
    → Optional: conversation topic tag (medical / casual / emergency / etc.)
    → Context embedding appended to refined_embedding
    ↓
[7] Translation Head
    → Autoregressive decoder (small, 2-layer Transformer decoder)
    → Produces top-K token candidates with per-token log-probability
    ↓
[8] Confidence Estimator
    → Computes sequence-level confidence score from:
       (a) max token probability mean across sequence
       (b) entropy of top-K distribution
       (c) cosine similarity to nearest neighbor in user correction memory
    → Three tiers:
       HIGH  (≥ 0.75) → proceed to output
       MED   (0.50–0.74) → generate disambiguation prompt
       LOW   (< 0.50) → request re-sign
    ↓
[9A] HIGH confidence → Communication Decision Layer
    → Final text output
    → On-device TTS (VITS or piper-tts, ARM-native)
    → Audio played through speakers
    ↓ ↑
[9B] MED confidence → Disambiguation UI
    → Display top-2 interpretations: "Did you mean: (A) [text1] or (B) [text2]?"
    → User selects via button or brief gesture
    → Selection stored as correction if it was not the top-1 prediction
    ↓
[9C] LOW confidence → Re-sign Request
    → Animated visual prompt on screen: "Please sign that again slowly"
    → Buffer resets, process repeats
    ↓
[10] User Correction Memory
    → If user overrides prediction: store (sequence_embedding → correct_text) pair
    → Used as soft retrieval override in future sessions (k-nearest-neighbor lookup)
    → Stored encrypted in local SQLite DB (AES-256)

┌─────────────────────────────────────────────────────────────┐
│                 HEARING PARTNER SIDE (Two-Way)               │
└─────────────────────────────────────────────────────────────┘

[11] Microphone captures partner's speech
     ↓
[12] Distil-Whisper (quantized, HTP backend) → real-time transcript
     ↓
[13] Text displayed in large font on the main SignBridge window
     (Deaf user reads the response)
     ↓
[14] Optional: Phi-3.5-Mini (on-device, GenieX runtime) summarizes
     long speech into key points if utterance > 3 sentences
```

---

## SECTION 8 — System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     SNAPDRAGON X ELITE / HP PC                        │
│                                                                        │
│  ┌──────────────┐    ┌──────────────────────────────────────────────┐ │
│  │   CAMERA     │───▶│  PERCEPTION MODULE (NPU — Hexagon HTP)       │ │
│  │  (built-in)  │    │  MediaPipe Holistic (quantized FP16)         │ │
│  └──────────────┘    │  Output: 225 landmarks/frame @ 30FPS          │ │
│                      └──────────────────┬───────────────────────────┘ │
│                                         │                              │
│                      ┌──────────────────▼───────────────────────────┐ │
│                      │  TEMPORAL ENCODER (NPU / CPU hybrid)          │ │
│                      │  4-layer Transformer, d=256                   │ │
│                      │  Sliding window: 64 frames (~2.1s)            │ │
│                      │  Output: sequence_embedding [256]             │ │
│                      └──────────────────┬───────────────────────────┘ │
│                                         │                              │
│  ┌──────────────────┐  ┌───────────────▼───────────────────────────┐ │
│  │ USER PROFILE DB  │─▶│  PERSONALIZATION LAYER (CPU)              │ │
│  │ (local SQLite,   │  │  User embedding [64] + sequence [256]     │ │
│  │  AES-256)        │  │  → refined_embedding [256]                │ │
│  └──────────────────┘  └──────────────────┬───────────────────────┘ │
│                                            │                           │
│  ┌──────────────────┐  ┌──────────────────▼──────────────────────┐  │
│  │ CONTEXT WINDOW   │─▶│  CONTEXT ENGINE (CPU)                   │  │
│  │ (last 5 turns,   │  │  Conversation history + topic tag       │  │
│  │  topic tag)      │  │  → context-refined embedding             │  │
│  └──────────────────┘  └────────────────────┬────────────────────┘  │
│                                              │                         │
│                         ┌────────────────────▼────────────────────┐  │
│                         │  TRANSLATION HEAD + CONFIDENCE (CPU/NPU) │  │
│                         │  2-layer Transformer decoder              │  │
│                         │  Top-K output + entropy score             │  │
│                         │  → HIGH / MED / LOW decision              │  │
│                         └────────────┬────────────────────────────┘  │
│                                      │                                 │
│          ┌───────────────────────────┼──────────────────────┐        │
│          ▼                           ▼                        ▼        │
│   ┌─────────────┐         ┌─────────────────┐      ┌──────────────┐  │
│   │  TTS OUTPUT │         │ DISAMBIGUATION  │      │  RE-SIGN UI  │  │
│   │ (piper-tts, │         │     DIALOG      │      │   PROMPT     │  │
│   │  ARM-native)│         └────────┬────────┘      └──────────────┘  │
│   └─────────────┘                  │                                   │
│                                    ▼                                   │
│                         ┌──────────────────────┐                      │
│                         │  CORRECTION MEMORY   │                      │
│                         │  (kNN store, SQLite) │                      │
│                         └──────────────────────┘                      │
│                                                                        │
│  ┌──────────────┐    ┌──────────────────────────────────────────────┐ │
│  │  MICROPHONE  │───▶│  DISTIL-WHISPER (NPU — Hexagon HTP)          │ │
│  │  (built-in)  │    │  Real-time ASR → transcript text             │ │
│  └──────────────┘    └──────────────────┬───────────────────────────┘ │
│                                         │                              │
│                      ┌──────────────────▼───────────────────────────┐ │
│                      │  DISPLAY OUTPUT (UI — Windows Native)         │ │
│                      │  Large-text partner speech display            │ │
│                      │  Optional: Phi-3.5 summarization (GenieX)    │ │
│                      └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

**Compute unit assignment:**

| Component | Unit | Reason |
|-----------|------|--------|
| MediaPipe Holistic landmark extraction | NPU (HTP FP16) | Proven on Qualcomm AI Hub, 52–76 µs/frame |
| Temporal Transformer encoder | NPU / CPU split | Encoder layers quantizable to INT8/FP16 |
| Personalization MLP | CPU | Small, fast, user-profile I/O required |
| Context engine | CPU | String/embedding ops, low compute |
| Translation decoder | CPU (+ NPU where ONNX Runtime offloads) | Autoregressive decode, sequential |
| Confidence estimator | CPU | Entropy calculation, low cost |
| Distil-Whisper ASR | NPU (HTP FP16) | Available on Qualcomm AI Hub |
| piper-tts / TTS | CPU | ARM-compiled, lightweight |
| Phi-3.5-Mini (optional) | NPU via GenieX | On-demand, not in critical path |
| Correction memory kNN | CPU | SQLite + numpy, low latency |

---

## SECTION 9 — AI Models Required

| Role | Model | Size | Format |
|------|-------|------|--------|
| Hand/body/face landmark extraction | MediaPipe Holistic | ~12 MB combined | ONNX / QNN |
| Temporal sequence encoder | Custom Transformer (4L, d=256) | ~8 MB | ONNX INT8 |
| Translation decoder | Custom Transformer (2L, d=256) | ~4 MB | ONNX INT8 |
| Personalization MLP | Custom (2-layer) | < 1 MB | ONNX |
| Confidence estimator | Statistical + small MLP | < 0.5 MB | NumPy/ONNX |
| ASR (hearing partner speech) | Distil-Whisper Small EN | ~240 MB | ONNX FP16 |
| TTS (speaking translation) | piper-tts (en_US-lessac-medium) | ~65 MB | ONNX |
| Summarization (optional) | Phi-3.5-Mini 4K (4-bit quant) | ~2.4 GB | GenieX/GGUF |

Total core pipeline (without optional Phi): approximately 330 MB. Comfortably fits in 8 GB RAM.

---

## SECTION 10 — Qualcomm AI Hub / Open-Source Models

| Model | Source | Status |
|-------|--------|--------|
| MediaPipe-Hand-Gesture-Recognition | [Qualcomm AI Hub](https://aihub.qualcomm.com/models/mediapipe_hand_gesture) | Available, HTP backend, ~52 µs inference |
| MediaPipe-Pose-Estimation | [Qualcomm AI Hub](https://aihub.qualcomm.com/models/mediapipe_pose) | Available |
| Distil-Whisper Small EN | [Qualcomm AI Hub](https://aihub.qualcomm.com/models/distil_whisper) | Available, HTP backend |
| Whisper-Base | [Qualcomm AI Hub / HuggingFace qualcomm/Whisper-Base](https://huggingface.co/qualcomm/Whisper-Base) | Available |
| Phi-3.5-Mini | GenieX / ONNX Runtime GenAI | Available for Snapdragon |
| piper-tts | Open-source (rhasspy/piper), ARM-compiled | Available |
| WLASL dataset | [WLASL GitHub (Bill Yuchen Lin)](https://github.com/dxli94/WLASL) | Public, CC-BY license |
| PHOENIX-2014T | [RWTH Aachen](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) | Public research license |
| How2Sign | Carnegie Mellon / UPC | Public research license |

**Custom models** (trained by team, not downloaded):
- Temporal Transformer encoder — trained on WLASL + PHOENIX + custom data
- Translation decoder — trained on same
- Personalization MLP — trained on enrollment data per user

---

## SECTION 11 — Snapdragon Optimization Strategy

### Why NPU Matters Here (Technical, Not Marketing)

The core bottleneck in any real-time sign-language system is the vision pipeline. At 30 FPS, the system has 33 ms per frame to extract landmarks. On a CPU alone, MediaPipe Holistic takes ~80–120 ms per frame on mid-range hardware — exceeding the budget and causing dropped frames. On the Snapdragon Hexagon NPU with HTP backend in FP16, Qualcomm AI Hub benchmarks show landmark extraction completing in 52–76 µs, which is ~1000× faster than the 33 ms budget. This headroom allows the temporal encoder and decoder to share CPU cycles without contention.

### Specific Optimizations

1. **Quantization**: MediaPipe landmark models deployed at FP16 on HTP. Custom Transformer encoder/decoder converted to INT8 using Qualcomm AI Model Efficiency Toolkit (AIMET) or ONNX quantization tools.

2. **ONNX Runtime with QNN Execution Provider**: The entire inference graph uses `onnxruntime` with `QNNExecutionProvider` targeting the Hexagon HTP. Configuration: `htp_performance_mode = burst`, `htp_graph_finalization_optimization_mode = 3`, `enable_htp_fp16_precision = 1`.

3. **Pipeline parallelism**: Frame capture and landmark extraction on NPU run as a background thread. Temporal encoder processes buffered landmark sequences independently. TTS synthesis is pre-queued while the decoder is still producing tokens. No step blocks another.

4. **Memory**: The full core pipeline fits in 330 MB model weights. With activations and buffers, total memory usage target is under 600 MB. No model paging required.

5. **Power**: The NPU's per-operation power consumption is significantly lower than running equivalent workloads on the CPU. The landmark extraction loop — which runs continuously — benefits most. This is measurable: the Snapdragon X Elite's NPU is documented to deliver >40 TOPS at a fraction of the CPU power draw.

6. **Distil-Whisper** on HTP: Running Distil-Whisper Small on the NPU achieves real-time factor < 0.1 (transcribes 10× faster than real speech), with low power cost.

### Model Conversion Path

```
PyTorch training → ONNX export → AIMET quantization → QNN compilation → .dlc / .bin
MediaPipe (already available in QNN format from Qualcomm AI Hub)
```

### Why This Needs a Snapdragon PC, Not a Cloud Server

| Dimension | Cloud approach | SignBridge (Snapdragon local) |
|-----------|---------------|-------------------------------|
| Latency | Network RTT + server queue (200–800 ms typical) | < 100 ms end-to-end on-device |
| Privacy | Raw landmarks or video transmitted | Zero data leaves device |
| Offline | Not possible | Fully functional |
| Cost | API/cloud fees per session | One-time device, no recurring cost |
| Medical/legal setting | Consent/HIPAA concerns for cloud | No cloud = no compliance issue |
| Battery | Network radio active continuously | NPU more power-efficient than LTE/WiFi constant stream |

---

## SECTION 12 — Hardware / Software Requirements

### Hardware (Target)
- HP Snapdragon X Elite Copilot+ PC (e.g., HP OmniBook X 14, HP EliteBook Ultra)
- Built-in RGB camera (720p or 1080p)
- Built-in microphone array
- Built-in speakers
- 16 GB RAM (8 GB minimum with core pipeline only)
- Snapdragon X Elite SoC with Hexagon NPU (45 TOPS)

### Software Stack
- Windows 11 (ARM64, Copilot+ edition)
- Python 3.11 (ARM64 native via Anaconda/miniforge)
- ONNX Runtime 1.18+ with QNN Execution Provider
- Qualcomm AI Hub Python SDK (`qai-hub`)
- OpenCV (ARM64 build, for camera capture)
- MediaPipe (landmark models from Qualcomm AI Hub)
- PyTorch 2.x (for training, not needed at inference)
- piper-tts (ARM64 binary)
- SQLite (standard library)
- PyQt6 or Tkinter (UI)
- cryptography (Python package, for AES-256 local storage)

### Optional
- GenieX runtime (for Phi-3.5-Mini summarization)
- AIMET (Qualcomm model quantization toolkit)

---

## SECTION 13 — Dataset Strategy

### Existing Datasets — Honest Assessment

| Dataset | Signs | Signers | Continuous? | License | Limitations |
|---------|-------|---------|-------------|---------|-------------|
| WLASL | 2,000 ASL words | 119 | Isolated only | CC-BY | Isolated signs, limited signer diversity, no facial detail |
| PHOENIX-2014T | ~1,200 weather vocab | 9 | Yes (weather domain) | Research | Extremely narrow domain, German SL |
| How2Sign | ~16,000 sentence segments | 10 | Yes | Research | English/ASL, uncontrolled background |
| ASL-LEX | ~2,700 signs | Few | Isolated | Research | Lexical resource, not video |
| MS-ASL | 1,000 signs | 200+ | Isolated | Research | Isolated signs |
| CSL-Daily | ~2,000 Chinese SL | Multiple | Yes | Research | Chinese SL only |

**Honest conclusion**: No existing dataset is sufficient for a deployable, personalized, context-aware system. WLASL and PHOENIX provide backbone pre-training. A custom dataset is required for the prototype vocabulary.

### Custom Prototype Dataset

The prototype focuses on **150 high-priority communication scenarios** rather than attempting full ASL. This makes the dataset buildable by a 2–3 person student team in 4–6 weeks.

**Vocabulary selection rationale**: chosen to cover the highest-frequency communication failures Deaf individuals report in public settings (based on Deaf accessibility research).

Categories:
- Emergency (15 signs): help, emergency, pain, doctor, police, fire, fall, medicine, allergy, phone
- Medical (25 signs): hurt, head, chest, stomach, breathing, fever, water, sick, dizzy, prescription, appointment, yes, no, name, address, birthday, repeat, slow, understand, not understand, thank you, please, need, want, where
- Daily interaction (40 signs): hello, goodbye, my name is, I don't understand, can you write that, phone number, price, pay, food, bathroom, wait, come back, quiet, loud, sit, stand, morning, afternoon, urgent
- Questions & responses (20 signs): what, who, when, where, why, how, yes, no, maybe, agree, disagree, correct, wrong, again, more, less, finished
- Connectors / grammar (30 signs): I/me, you, he/she, we, they, is/am/are, not, also, before, after, now, later, here, there, today, tomorrow, already, still, this, that
- Fingerspelling support (26 letters)

**Collection protocol per sign:**
- 8 participants (3 Deaf/HoH signers, 5 hearing learners with varied proficiency)
- 5 repetitions per signer per sign
- 3 background conditions (plain wall, cluttered, outdoor)
- 2 lighting conditions (bright, dim)
- 2 hand dominance options (left-handed and right-handed variants)
- 3 signing speeds (normal, slow, fast)
- Continuous sequences: 200 recorded 3–7 sign sentences per participant
- Total estimated clips: ~8,000 isolated + ~1,600 sentence clips

**Total dataset size**: approximately 10,000 clips, manageable for a 3-person team.

### Pre-training Strategy
1. Pre-train temporal encoder on WLASL (2,000 signs, backbone features)
2. Fine-tune on custom 150-sign dataset
3. Personalization layer trained separately per user during 5-minute enrollment

---

## SECTION 14 — Personalization Mechanism

### Problem
Every person signs differently. Speed varies. Hand size affects landmark distances. Users with motor differences, older signers, or regional dialect users produce signs that deviate from training data. A one-size-fits-all model will systematically fail these users.

### Solution: User Embeddings + Lightweight Adapter

**Enrollment (5 minutes, first time):**
1. User signs 30 reference signs from a guided on-screen prompt (covers key hand configurations)
2. System extracts temporal embeddings for each reference sign
3. A 64-dimensional user profile vector is computed by averaging and PCA-projecting the embeddings
4. Stored locally: `user_profile.json` (encrypted)

**Inference personalization:**
- The user embedding (64-dim) is concatenated with every sequence embedding (256-dim)
- A 2-layer MLP (320 → 256) learns to adjust the embedding space per user
- This MLP is the personalization adapter — it is lightweight (< 500K parameters) and can be fine-tuned on 30 reference signs in under 60 seconds on CPU

**Continual correction:**
- Every time the user corrects a prediction, the (embedding, correct_label) pair is added to a local kNN correction store
- At inference, after the translation head produces output, the top-1 correction match (if cosine similarity > 0.85) overrides the output
- This is retrieval-augmented personalization, not retraining — no GPU needed

**Privacy**: all personalization data is stored locally, AES-256 encrypted. No biometric data transmitted.

---

## SECTION 15 — Context-Aware Mechanism

### Problem
Sign language, like spoken language, is highly context-dependent. The same sign can mean different things in different conversations. Example: the ASL sign for "medicine" can also overlap with "drug" — in a medical context, "medicine" is correct; in a street context, "drug" might be intended. Without context, the model always picks the statistically most frequent interpretation.

### Solution: Sliding Context Window + Topic Tag

**Conversation context window:**
- Stores the last 5 translated utterances as a string list
- Computes a simple TF-IDF topic vector from the accumulated conversation
- This topic vector is embedded and concatenated with the sequence embedding before decoding

**Topic tag:**
- User selects a session topic at launch (Medical / General / Emergency / Work / Education)
- Topic tag biases the decoder's vocabulary distribution (implemented as a learned topic embedding added to the decoder input)
- Example: in Medical mode, "medicine" is up-weighted vs. "drug"

**Disambiguation using context:**
- When confidence is in the MED tier, instead of just showing "Did you mean A or B?", the system uses the conversation history to rank which interpretation is more likely and presents that one first

**Limitation (honest):** The context engine is not a full conversational language model. It is a topic-conditioned bias mechanism. A future version could use on-device Phi-3.5-Mini to do true conversational reasoning, but for the MVP the sliding window + topic tag is achievable and demonstrably useful.

---

## SECTION 16 — Uncertainty Handling

This is one of the clearest differentiators from every existing system.

### Why It Matters
Sign language has genuine ambiguity. A 2024 study (Reconsidering Sentence-Level SLT, arxiv 2406.11049) found that for 33% of sentences, even fluent Deaf signers needed additional discourse context to understand a clip. Pretending 100% confidence is dishonest and dangerous in medical or emergency contexts.

### Three-Tier Confidence System

```
Score ≥ 0.75 → HIGH CONFIDENCE
  Action: Translate and speak immediately
  Display: Text + audio, no interruption to user

0.50 ≤ Score < 0.75 → MEDIUM CONFIDENCE
  Action: Show disambiguation dialog
  Display: "I think you said: [Option A] or [Option B]?"
  User selects by button tap (or brief head nod detected via pose)
  Result: Correct output + correction stored if model was wrong

Score < 0.50 → LOW CONFIDENCE
  Action: Request re-sign
  Display: Animated icon + text: "Please sign that again"
  No audio output (prevents wrong speech to hearing partner)
  Buffer resets, system re-processes
```

### Confidence Score Computation

```
confidence = w1 × mean_token_prob + w2 × (1 - entropy_norm) + w3 × knn_match_score
```

Where:
- `mean_token_prob`: average of per-token softmax probabilities across the decoded sequence
- `entropy_norm`: normalized entropy of the top-K distribution (high entropy = low confidence)
- `knn_match_score`: cosine similarity to the nearest neighbor in the correction memory (if user has seen this sign before and corrected it, confidence in that correction is high)
- `w1, w2, w3 = 0.5, 0.3, 0.2` (tuneable hyperparameters)

---

## SECTION 17 — Privacy Strategy

Privacy is an architectural constraint, not a feature checkbox.

| Principle | Implementation |
|-----------|----------------|
| Raw frames never stored | Camera frames processed in a circular RAM buffer of 64 frames. Never written to disk. |
| Landmarks not transmitted | All inference is local. No network calls in the core pipeline. |
| User profile encrypted | AES-256 via Python `cryptography` library. Key derived from Windows DPAPI per-user. |
| Correction memory encrypted | Same SQLite database, same encryption. |
| Conversation history | Stored only in RAM during session. Cleared on app close (configurable to persist for context continuity with user consent). |
| Optional telemetry | Disabled by default. If user opts in, only aggregate accuracy metrics (no text, no video) are collected. |
| GDPR/compliance posture | All data local. No personal data transmitted. User can delete profile + correction memory in one click from Settings. |

**Why this matters specifically for the target use cases:** Medical conversations between a Deaf patient and a doctor contain legally protected health information. Any cloud transmission would require HIPAA compliance, BAAs with cloud providers, and patient consent. Running entirely on-device eliminates all of these requirements.

---

## SECTION 18 — Offline Strategy

### What Runs Offline (Core Pipeline — 100% Offline)
- MediaPipe landmark extraction
- Temporal Transformer encoder/decoder
- Personalization layer
- Context engine
- Confidence estimator
- Distil-Whisper ASR
- piper-tts
- Correction memory retrieval
- User profile management

**Every feature that matters for real-time communication works with zero network connectivity.**

### What Requires Network (Optional Features Only)
- Model updates (downloaded once, applied offline)
- Optional: sending anonymized accuracy feedback (opt-in, async)
- Optional: Phi-3.5-Mini initial download (model is large, downloaded once)

### Why This Is Not Just a Marketing Claim
The models are explicitly chosen to fit on-device:
- Total core model weights: ~330 MB
- Target RAM footprint: < 600 MB
- All models in ONNX format, runnable without network
- Distil-Whisper and piper-tts are both proven offline-capable

This is fundamentally different from SL2T, which sends coordinates to a Google server.

---

## SECTION 19 — User Feedback Mechanism

### In-Session Correction
1. After translation is spoken, the translated text appears on screen
2. A small "Incorrect" button is always visible
3. User taps "Incorrect" → text input or quick-select menu appears with alternatives
4. User selects or types the correct translation
5. System stores (sequence_embedding, correct_text) pair in local kNN correction memory
6. UI confirms: "Got it. I'll remember that."

### Cross-Session Learning
The correction memory persists across sessions (encrypted SQLite). At inference:
1. Current sequence embedding is computed
2. kNN lookup against stored corrections (cosine similarity, top-1)
3. If similarity > 0.85 and confidence tier was MED or LOW, correction override is applied
4. The correction's confidence contribution boosts the score for this user

### Limits (Honest)
- This is retrieval-based, not true continual learning. The base model does not retrain.
- If two corrections conflict (same embedding maps to different texts), the more recent one is preferred.
- The kNN store grows unbounded in theory; a pruning policy (max 500 corrections, LRU eviction) is applied.

---

## SECTION 20 — MVP Features

The following 7 features are realistic for a 2–3 student team in 8–10 weeks and together demonstrate the core differentiation:

| # | Feature | Why This Makes the Cut |
|---|---------|----------------------|
| 1 | Real-time hand + upper-body landmark extraction at 30 FPS on Snapdragon NPU | Foundation — proves Snapdragon optimization |
| 2 | Continuous sign sequence recognition (not isolated signs) for 150-sign vocabulary | Proves non-trivial recognition |
| 3 | Three-tier confidence system with disambiguation dialog and re-sign request | The clearest differentiator from every existing system |
| 4 | User enrollment + personalization (5-minute setup, per-user embeddings) | Second strongest differentiator |
| 5 | Text-to-speech output (translated text spoken aloud) | Completes the communication bridge to hearing partner |
| 6 | Speech-to-text (hearing partner's speech → displayed text for Deaf user) | Enables two-way communication |
| 7 | User correction memory (persistent, cross-session, kNN retrieval) | Demonstrates learning without retraining |

**Not in MVP (cut for feasibility):**
- Phi-3.5-Mini summarization (optional add-on)
- Emergency mode (adaptive communication mode)
- More than 150 signs in the custom vocabulary
- Sign production/animation for the hearing partner

---

## SECTION 21 — Advanced Features (Post-MVP / Future)

| Feature | Description | Feasibility |
|---------|-------------|-------------|
| Full ASL vocabulary expansion | Extend from 150 to 1,000+ signs using WLASL + transfer learning | Medium — data bottleneck |
| Adaptive communication mode | Emergency mode (speed priority), repeated-misunderstanding mode (pause, explain) | Medium |
| Sign production output | Display animated avatar signing the hearing partner's speech for the Deaf user | Hard — requires separate generative model |
| Multi-language support | BSL, ISL, Auslan, etc. | Hard — requires new datasets |
| Phi-3.5-Mini integration | Contextual summarization, intent reasoning | Easy — model already available on Snapdragon |
| Wearable companion | Pair with a smartwatch for vibration alerts when confidence is LOW | Medium |
| Federated personalization | Improve base model from anonymized correction signals across users | Hard — privacy engineering required |

---

## SECTION 22 — 2–3 Minute Demo Script

**Setup**: Laptop (HP Snapdragon PC) on a table. One person playing a Deaf signer. One person playing a hearing doctor/pharmacist. The app is open in full-screen mode showing a clean, accessible UI. Wi-Fi is turned OFF to demonstrate offline operation.

---

**[0:00 – 0:15] The Problem**
> Presenter: "Imagine you're Deaf and you walk into a pharmacy. The pharmacist doesn't know sign language. There's no interpreter available. What do you do? This is SignBridge."

---

**[0:15 – 0:45] Enrollment & Personalization**
> The Deaf user (Demo Person A) is shown doing a 5-minute enrollment — the presenter skips this live and shows a pre-recorded clip of it, then says: "After a 5-minute one-time setup, SignBridge knows how *you* sign — not just how the average person does."

---

**[0:45 – 1:10] Signing → Speech (High Confidence)**
> Demo Person A faces the camera and signs a 4-sign sequence: MEDICINE + NEED + MORNING + QUESTION  
> SignBridge processes in real time. Landmarks light up on screen. After ~1.5 seconds:  
> **Text appears: "Do I need to take the medicine in the morning?"**  
> **TTS audio plays aloud.**  
> Presenter: "The system understood a full intent from a continuous sign sequence — not one sign at a time."

---

**[1:10 – 1:30] Ambiguity Handling (Medium Confidence)**
> Demo Person A signs a shorter, faster sequence that is ambiguous. A dialog appears:  
> **"Did you mean: (A) 'I have an allergy' or (B) 'I have a reaction'?"**  
> Demo Person A taps A.  
> System speaks: *"I have an allergy."*  
> Presenter: "When the system isn't sure, it asks — it doesn't guess silently."

---

**[1:30 – 1:50] Two-Way Communication**
> Demo Person B (pharmacist) speaks: "Which medication are you referring to?"  
> Distil-Whisper transcribes in real time — large text appears on screen for Demo Person A.  
> Presenter: "The hearing partner's speech is transcribed and shown immediately. This is a two-way bridge."

---

**[1:50 – 2:05] Correction Learning**
> Demo Person A signs a sign that was mis-translated. They tap "Incorrect" and type the correct meaning.  
> **"Got it. I'll remember that for next time."**  
> Presenter: "SignBridge learns from corrections — and that memory is stored entirely on your device, private to you."

---

**[2:05 – 2:15] Offline Proof**
> Presenter holds up the laptop, opens network settings, shows Wi-Fi is OFF.  
> Signs another phrase. It works.  
> **"Fully offline. Your conversation never leaves this device."**

---

**[2:15 – 2:30] Close**
> Presenter: "SignBridge runs entirely on a Snapdragon-powered HP PC — no cloud, no subscription, no privacy risk. It adapts to you, it handles ambiguity honestly, and it bridges both sides of the conversation. That's the gap we're closing."

---

## SECTION 23 — Competitor Comparison Table

| Product / System | Continuous Signing | Context Awareness | Personalization | Two-Way | Offline | On-Device AI | Privacy (no cloud) | Uncertainty Handling | User Feedback Learning | Platform |
|---|---|---|---|---|---|---|---|---|---|---|
| **Google SL2T** (Aug 2026) | ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No (server-side translation) | Partial (landmarks on-device, translation on server) | ❌ Landmarks sent to server | ❌ No | ❌ No | Android/Pixel 11 only |
| **Signapse SignStream** | ❌ Text→Sign only | ❌ No | ❌ No | ❌ No (broadcast only) | ❌ No | ❌ Cloud API | ❌ Cloud | ❌ No | ❌ No | Web API |
| **SignAll** | Not clearly documented | ❌ No | Not clearly documented | Partial | ❌ No | ❌ Cloud | ❌ Cloud | ❌ No | ❌ No | Proprietary hardware |
| **Spotter+GPT** (research) | Partial (spotting + LLM) | Partial (LLM context) | ❌ No | ❌ No | ❌ No | ❌ Cloud LLM required | ❌ Cloud | ❌ No | ❌ No | Research only |
| **SHuBERT+ByT5 RT** (research 2026) | ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No | Partial (capture on edge, compute on server) | ❌ Server required | ❌ No | ❌ No | Research only |
| **Bidirectional SLT (IJERT 2026)** | ❌ Isolated only | ❌ No | ❌ No | Partial (sign→text + text→animation) | Not clearly documented | Not clearly documented | Not clearly documented | ❌ No | ❌ No | Research / mobile |
| **Few-Shot Prototypical (arxiv 2512)** | ❌ Isolated only | ❌ No | Partial (few-shot) | ❌ No | Potentially | CPU/GPU | Potentially | ❌ No | ❌ No | Research only |
| **Mobile SLR (Springer 2026)** | ✅ Yes (pre-recorded) | ❌ No | ❌ No | ❌ No | Partial | On-device | Partial | ❌ No | ❌ No | Mobile |
| **UPRet (arxiv 2405)** | N/A (retrieval) | ❌ No | ❌ No | ❌ No | Not deployed | Not deployed | N/A | Partial (model-level) | ❌ No | Research only |
| **Adaptive Simultaneous SLT (ACL 2024)** | ✅ Yes (streaming) | Partial (length estimation) | ❌ No | ❌ No | Not deployed | Not deployed | Not deployed | Partial (model output) | ❌ No | Research only |
| **SignBridge (proposed)** | ✅ Yes | ✅ Yes (topic+history) | ✅ Yes (5-min enrollment) | ✅ Yes | ✅ Yes | ✅ Full (100% on-device) | ✅ Yes (zero transmission) | ✅ Yes (3-tier UI) | ✅ Yes (kNN + AES store) | Windows/Snapdragon PC |

---

## SECTION 24 — Technical Novelty Analysis

### What Is Genuinely Novel

1. **Uncertainty-as-interaction-design for sign language**: UPRet models uncertainty at the retrieval embedding level as a research contribution. The ACL 2024 paper handles confident translation length. Neither exposes uncertainty as a three-tier user interaction loop — ask / request-re-sign / auto-translate — in a live communication system. This combination is novel as a deployed UX pattern.

2. **Persistent user correction memory with kNN retrieval in a sign language system**: No reviewed system (commercial or research) implements cross-session user correction persistence with embedding-space retrieval override. The closest is general-purpose speech recognition adaptation, but that is a completely different domain and modality.

3. **Full two-way offline communication bridge on a PC platform**: The combination of (sign→speech) + (speech→text for Deaf user) + (offline) + (Windows/Snapdragon) has no direct prior art. SL2T is one-way and cloud-dependent. Bidirectional IJERT paper is isolated-sign-only and not fully deployed.

4. **5-minute enrollment personalization without full retraining**: Few-shot prototypical networks have been studied in isolation. Applying them as a real-time personalization adapter within a running communication pipeline, combined with retrieval correction memory, is not documented in any reviewed work.

### What Is NOT Novel (Honesty Required)

- Continuous signing with a Transformer encoder: well-established (PHOENIX, SHuBERT, etc.)
- On-device landmark extraction: Google SL2T does this (130 landmarks)
- ASR integration: standard, Whisper is widely used
- TTS output: standard, piper-tts and similar widely used
- Context-conditioned decoding: present in some research (topic modeling in NLP is old)
- ONNX / Snapdragon NPU deployment: Qualcomm AI Hub provides this as infrastructure

### Honest Novelty Score: **7 / 10**

Justification: The system solves a real gap with a defensible technical architecture. No single component is groundbreaking. The novelty is in the integration: uncertainty-as-interaction + persistent correction memory + personalization + two-way offline bridge on Snapdragon PC — as a complete, privacy-first communication tool. This combination does not exist as a deployed or even prototyped system in the reviewed literature. The score is 7 and not 8 because: (a) each component has prior art in isolation, (b) the custom dataset is small and the vocabulary is limited, and (c) the translation accuracy on continuous signing at 150 signs will be imperfect. A score of 8 would require a larger dataset, a more novel model architecture, or a formal user study demonstrating real-world impact. The 7 is defensible and buildable.

---

## SECTION 25 — Risks and Limitations

| Risk | Severity | Probability |
|------|----------|-------------|
| Translation accuracy on continuous signing may be low (< 70% on sentence-level) | High | High — this is an inherently hard problem |
| 150-sign custom dataset too small to generalize well | High | Medium |
| Personalization enrollment insufficient for users with significant motor differences | Medium | Medium |
| MediaPipe landmark quality degrades in poor lighting | Medium | High — well-documented limitation |
| NPU model conversion introduces accuracy loss from quantization | Medium | Medium |
| piper-tts voice quality may sound robotic, reducing perceived professionalism | Low | Low — piper-tts is reasonably natural |
| Two-way mode: Distil-Whisper may miss words in noisy environments | Medium | Medium |
| User correction memory conflicts if sign is inconsistent across sessions | Low | Low |
| Demo hardware (Snapdragon PC) unavailable to all team members | High (for dev) | Medium |
| Scope creep during implementation | Medium | High |

---

## SECTION 26 — How to Overcome the Risks

| Risk | Mitigation |
|------|-----------|
| Low translation accuracy | Limit demo vocabulary to 50 most-practiced signs for the demo. Focus on sentence-level accuracy for those 50. Document accuracy honestly. |
| Small dataset | Augment with WLASL pre-training. Use data augmentation (landmark jitter, speed perturbation, horizontal flip for handedness). |
| Personalization limitations | Increase enrollment to 50 signs. Provide visual feedback during enrollment (quality indicator). Allow re-enrollment. |
| Poor lighting | Add a lighting quality indicator in the UI. Prompt user to improve lighting before starting. |
| Quantization accuracy loss | Use FP16 not INT8 for the translation decoder. Reserve INT8 for landmark extractor only. |
| TTS quality | Use piper-tts `high` quality model. Test voices and select best one for English. |
| Noisy ASR | Show confidence bars on the ASR transcript. Let user tap words to correct. |
| NPU hardware access | Use Qualcomm AI Hub's hosted device testing (cloud-accessible NPU simulation) during development. Test on actual hardware for final demo. |
| Scope creep | Follow the MVP list strictly. Advanced features are documented but not implemented unless MVP is complete. |

---

## SECTION 27 — Evaluation Metrics

A meaningful evaluation goes beyond accuracy.

| Metric | How Measured | Target |
|--------|-------------|--------|
| Sign recognition accuracy (isolated, top-1) | % correct on 150-sign test set | > 80% |
| Sentence-level intent accuracy | Human evaluation: does output match intended meaning? (judge rating 1–5) | Mean ≥ 3.5/5 |
| Word Error Rate (WER) | Against ground-truth text for continuous signing test set | < 40% for 50-sign focused vocab |
| False positive rate (wrong translation spoken) | % utterances where wrong speech was produced | < 10% |
| False negative rate (re-sign requested when clear sign was given) | % clear signs that triggered LOW confidence | < 15% |
| Disambiguation precision | % of MED-confidence cases where top-2 contained correct answer | > 85% |
| End-to-end latency (sign-end to speech-start) | Measured with timestamps: last frame of sign → first audio sample | < 1.5 seconds |
| FPS (landmark extraction) | OpenCV frame counter | ≥ 25 FPS |
| NPU utilization | Windows Task Manager / Qualcomm AI Hub profiler | > 30% NPU load during landmark phase |
| RAM usage | Process memory during inference | < 600 MB |
| ASR Word Accuracy (hearing partner side) | Standard WER on quiet speech | > 90% |
| Personalization improvement | Accuracy before enrollment vs. after enrollment for same user | ≥ +10% |
| Offline functionality | All features tested with no network | 100% |
| User correction recall | % of corrections successfully recalled in next session | > 90% |
| Cross-user accuracy | Base model accuracy on unseen users (no enrollment) | Document baseline honestly |

---

## SECTION 28 — Development Roadmap

### Phase 1 — Foundation (Weeks 1–2)
- Set up Windows ARM64 Python environment with ONNX Runtime + QNN provider
- Verify MediaPipe Holistic landmark extraction running on Snapdragon NPU (use Qualcomm AI Hub hosted device if needed)
- Verify Distil-Whisper running on NPU
- Set up piper-tts
- Build basic camera → landmark → display pipeline in Python

### Phase 2 — Data Collection (Weeks 2–4, parallel with Phase 1)
- Record custom 150-sign dataset per the protocol in Section 13
- Pre-process: landmark extraction, normalization, sequence alignment
- Prepare WLASL pre-training data (download, extract landmarks, store)

### Phase 3 — Model Training (Weeks 3–5)
- Pre-train temporal Transformer encoder on WLASL
- Fine-tune on custom 150-sign data
- Train translation decoder (encoder → text)
- Train personalization MLP (on enrollment examples)
- Export all models to ONNX
- Quantize and validate on Snapdragon NPU

### Phase 4 — Integration (Weeks 5–7)
- Connect full pipeline: camera → landmarks → encoder → decoder → TTS
- Implement confidence estimator
- Implement disambiguation UI and re-sign request UI
- Implement user enrollment flow
- Implement correction memory (SQLite + kNN)
- Implement two-way mode (Whisper → text display)
- Implement context window and topic tag

### Phase 5 — Testing & Evaluation (Weeks 7–8)
- Measure all metrics from Section 27
- Fix accuracy issues for demo vocabulary
- Test on multiple users
- Test offline
- Optimize latency

### Phase 6 — Presentation & Demo Polish (Weeks 8–10)
- Write demo script, practice, time it
- Finalize UI
- Prepare slides
- Record backup video of working demo
- Document everything in this repo

---

## SECTION 29 — Team Member Responsibilities (2–3 Students)

### For a 2-Person Team

**Person A — AI / ML Engineer**
- Data collection and pre-processing pipeline
- Temporal Transformer encoder/decoder training
- Personalization MLP training
- ONNX export and Snapdragon NPU quantization
- Confidence estimator implementation
- kNN correction memory implementation
- Model evaluation and metric measurement

**Person B — Systems / Application Engineer**
- Windows ARM64 environment setup
- Camera capture and frame buffer pipeline
- Integration of all AI models into unified pipeline
- UI development (PyQt6 / Tkinter)
- Distil-Whisper integration (two-way mode)
- piper-tts integration
- SQLite encrypted storage
- Demo script and presentation

### For a 3-Person Team (add)

**Person C — Data & Evaluation**
- Lead custom dataset recording
- Dataset pre-processing and quality control
- Evaluation metric measurement and reporting
- Research documentation (this document)
- Presentation slides and demo video

---

## SECTION 30 — Estimated Implementation Difficulty

| Component | Difficulty | Reason |
|-----------|-----------|--------|
| Environment setup (Windows ARM64 + ONNX + QNN) | Medium | Well-documented, some ARM64 quirks |
| MediaPipe landmark extraction on NPU | Easy | Available directly from Qualcomm AI Hub |
| Distil-Whisper on NPU | Easy | Available from Qualcomm AI Hub, straightforward |
| piper-tts integration | Easy | Well-documented, ARM64 binaries available |
| Custom dataset recording | Medium | Time-consuming, quality control needed |
| Temporal Transformer training | Hard | Requires ML knowledge, GPU for training, careful data prep |
| ONNX export + NPU quantization | Medium | Well-documented toolchain, some debugging expected |
| Confidence estimator | Easy-Medium | Statistical + small MLP, straightforward once model is trained |
| Personalization MLP | Medium | Clear architecture, requires careful enrollment UX |
| kNN correction memory | Easy | NumPy + SQLite, well-understood pattern |
| Context engine | Easy | String processing + simple embedding concat |
| UI (PyQt6) | Medium | Not trivial, but well-documented |
| End-to-end integration | Hard | Synchronizing async pipeline, debugging latency |
| **Overall difficulty for 2–3 students in 10 weeks** | **7/10** | Achievable with discipline and scope control |

---

## SECTION 31 — What to Show in the Final Presentation

### Slide deck structure (10–12 slides, 5 minutes)

1. **The Problem** — Real photo/scenario of a Deaf person at a pharmacy/hospital. One powerful statistic.
2. **Why Existing Solutions Fail** — Competitor comparison table (Section 23). Highlight SL2T's launch and its gaps.
3. **SignBridge: The Concept** — One-line concept. Architecture diagram (simplified).
4. **The Three Differentiators** — (1) Uncertainty interaction, (2) Personalization, (3) Two-way offline bridge. Each in one bullet.
5. **Snapdragon: Why It Matters** — The table from Section 11 comparing cloud vs. local. NPU utilization numbers.
6. **Live Demo** — 2.5 minutes (run demo from Section 22)
7. **Technical Architecture** — Simplified pipeline diagram with NPU/CPU workload breakdown
8. **Dataset & Training** — Honest description of 150-sign custom dataset
9. **Evaluation Results** — Key metrics table. Be honest about current accuracy.
10. **Privacy** — "Your conversation never leaves this device." Bullet points from Section 17.
11. **Roadmap** — What's in the MVP, what comes next
12. **Team + Abstract**

### Backup video
Always have a pre-recorded demo video in case live demo hardware fails.

---

## SECTION 32 — Competition Abstract (400 words)

**SignBridge: Private, Personalized, On-Device Communication Intelligence for Sign Language Users**

Every day, millions of Deaf and hard-of-hearing individuals face critical communication barriers when interacting with people who do not know sign language — at hospitals, pharmacies, workplaces, and government offices. Human interpreters are unavailable on demand. Writing is slow and assumes literacy in a spoken language that is not a sign language user's first language. Existing AI sign-language tools translate isolated gestures into text, require cloud connectivity, offer no adaptation to individual signing styles, and provide no mechanism to handle the genuine ambiguity inherent in sign language interpretation. They solve a part of the problem. SignBridge solves the problem.

SignBridge is a fully on-device, Snapdragon-optimized communication intelligence system that runs on HP Copilot+ PCs. It translates continuous sign language sequences into spoken speech in real time, and simultaneously transcribes the hearing partner's speech into large-text display for the Deaf user — creating a true two-way communication bridge that requires no internet connection, no cloud service, and no specialized hardware beyond the built-in camera and microphone.

Three capabilities differentiate SignBridge from every reviewed system. First, a three-tier uncertainty interaction system: rather than silently mistranslating ambiguous signs, SignBridge asks for clarification when confidence is moderate, and requests the sign be repeated when confidence is too low to act on. This is not a model-level detail — it is a communication safety feature. Second, a five-minute personal enrollment that adapts the system to each user's unique signing style using lightweight user embeddings, without any retraining. A signer with motor differences, a regional dialect, or an idiosyncratic signing pattern is served as well as any other user. Third, a persistent correction memory that stores the user's feedback in an encrypted local database and recalls it in future sessions — the system gets better the more it is used, without ever transmitting data.

The entire pipeline is powered by the Snapdragon Hexagon NPU. MediaPipe Holistic landmark extraction runs at under 76 µs per frame on the Hexagon HTP backend. Distil-Whisper provides real-time ASR at negligible CPU cost. All models run in ONNX format with QNN Execution Provider. No frame of video, no landmark coordinate, and no word of conversation leaves the device. For a medical consultation, a legal appointment, or an emergency situation, this is not a convenience — it is a requirement.

SignBridge demonstrates that the most important AI problem in accessibility is not achieving the highest BLEU score on a benchmark. It is building a system that works for real people, in real situations, without asking them to trust their most sensitive conversations to a server.

---

## SECTION 33 — Problem Statement

The 430 million people worldwide with disabling hearing loss who use sign language as a primary communication modality face a structurally unresolved communication gap when interacting with the hearing majority. This gap is most severe — and most dangerous — in high-stakes settings: medical consultations, emergency response, legal proceedings, and employment. In these contexts, professional interpreters are frequently unavailable, expensive, and logistically impractical. Digital solutions have focused on sign-to-text translation without addressing the full communication scenario: the hearing person also speaks, the signing person needs to receive that response accessibly, and the entire exchange must handle ambiguity, individual variation, and connectivity failure without degrading the communication. No currently deployed product or research prototype addresses this complete scenario in a private, offline, personalized manner. The problem is not a lack of AI capability — it is a lack of integrated design.

---

## SECTION 34 — Innovation Statement

SignBridge introduces three integrated innovations that, combined, have no direct prior art in deployed sign-language communication systems. The first is uncertainty-as-interaction: a three-tier confidence system that makes the AI's epistemic state visible and actionable to the user, replacing silent mistranslation with structured dialogue. The second is enrollment-based personalization: a five-minute setup that derives a user-specific embedding, enabling the system to recognize an individual's signing style without full retraining, with cross-session improvement driven by correction feedback stored in a local encrypted memory. The third is a complete two-way offline bridge: sign-to-speech for the signing user and speech-to-text for the hearing partner, operating entirely on the Snapdragon NPU with zero network dependency. Together, these three features address the communication gap not at the model benchmarking level, but at the real-world deployment level.

---

## SECTION 35 — "Why Snapdragon?" Statement

Sign-language communication is time-sensitive, spatially private, and frequently happens in connectivity-constrained environments — exactly the conditions that make cloud AI inadequate. Snapdragon's Hexagon NPU delivers the landmark extraction and ASR inference at sub-millisecond speeds that a real-time conversation demands, without the 200–800 ms round-trip latency of cloud processing. The privacy argument is equally concrete: a Deaf patient discussing a diagnosis, a signing employee in a confidential meeting, or a Deaf individual filing a legal complaint cannot and should not have their conversation represented in landmark coordinates being streamed to a server. The Snapdragon NPU enables full inference locally, keeping every word, sign, and facial marker within the physical device. Beyond latency and privacy, the Snapdragon platform enables continuous always-on operation with NPU-accelerated inference that consumes a fraction of the power that equivalent CPU workloads would require — making the system viable for extended real-world use. SignBridge is not optimized for Snapdragon as a marketing decision. It is architected around Snapdragon because the platform's specific properties — NPU throughput, offline capability, ARM-native inference stack, and Windows integration — are the precise technical requirements of the problem being solved.

---

## SECTION 36 — Judge-Perspective Assessment

**Honest scoring prediction across the four hackathon dimensions:**

**1. Technical Implementation (expected score: Strong)**
- The pipeline is technically credible. MediaPipe + Transformer + ONNX/QNN on Snapdragon is a real, buildable stack.
- The personalization mechanism is technically specific (user embeddings, MLP adapter, kNN correction memory) — not hand-waving.
- Quantization strategy, NPU workload assignment, and latency targets are precise.
- Risk: if the demo shows low translation accuracy, judges who know sign language AI will notice. Mitigation: limit demo vocabulary to 50 well-trained signs, be transparent about accuracy on the rest.

**2. Application Use Case & Innovation (expected score: Strong)**
- The problem is real and important. The communication gap in healthcare/emergency settings is well-documented.
- The three differentiators (uncertainty interaction, personalization, two-way offline bridge) are clearly explainable and demonstrably absent in competitors including the just-launched Google SL2T.
- Risk: judges may ask "Why not just use SL2T?" — answer is: SL2T is Android-only, cloud-dependent, one-way, and has no personalization. This answer is documented and rehearsed.

**3. Deployment & Accessibility (expected score: Good to Strong)**
- The system runs on available HP Snapdragon hardware, uses a standard Python stack, and has a clear installation path.
- The offline capability is demonstrable live (turn off Wi-Fi).
- Risk: a Windows Python app is not a polished consumer product. The UI should be clean and the demo should not show a terminal window.

**4. Presentation & Documentation (expected score: Strong)**
- This document provides complete, honest, research-backed documentation covering all 36 required areas.
- The demo script is specific, rehearsable, and designed to answer the key judge question: "What makes this different from a sign-language classifier?"
- Risk: the presentation must be delivered confidently. Technical depth is available; it must not overwhelm a non-technical judge in the room.

**Overall assessment**: SignBridge is a competition-level entry with a defensible, technically honest differentiation, a realistic MVP, and a clear connection to Snapdragon's specific capabilities. It is not the highest possible novelty (10/10 would require a published research advance). It is an honest, well-designed, buildable 7/10 system that addresses a real gap better than every currently deployed solution.

---

*Document prepared for the Snapdragon AI Lab Build & Present Challenge*  
*All prior-art claims are based on literature search conducted September 2026*  
*Content was rephrased for compliance with licensing restrictions where source material was cited*
