# SignBridge — Dataset Strategy

## Existing Datasets — Honest Assessment

| Dataset | Signs | Signers | Continuous | License | Key Limitation |
|---------|-------|---------|------------|---------|----------------|
| WLASL | 2,000 ASL | 119 | ❌ Isolated | CC-BY | Isolated only, limited signer diversity |
| PHOENIX-2014T | ~1,200 (weather) | 9 | ✅ Yes | Research | Narrow domain (weather), German SL |
| How2Sign | ~16,000 segments | 10 | ✅ Yes | Research | ASL only, some noise |
| MS-ASL | 1,000 signs | 200+ | ❌ Isolated | Research | Isolated signs only |
| ASL-LEX | ~2,700 signs | Few | ❌ Lexical | Research | No video |
| CSL-Daily | ~2,000 Chinese SL | Multiple | ✅ Yes | Research | Chinese SL only |

**Verdict**: No dataset covers the full SignBridge use case. WLASL is used for pre-training. A custom dataset is required for the prototype's 150-sign communication vocabulary.

---

## Custom Prototype Dataset

### Vocabulary — 150 Signs Across 5 Categories

| Category | Count | Rationale |
|----------|-------|-----------|
| Emergency | 15 | Highest-stakes communication failures (hospital, police, fire) |
| Medical | 25 | Most common Deaf healthcare communication barrier |
| Daily interaction | 40 | Real-world encounters (shop, office, transport) |
| Questions & responses | 20 | Grammar: disambiguation, clarification |
| Connectors / grammar | 24 | Sentence construction |
| Fingerspelling (A–Z) | 26 | Names, technical terms |

---

### Collection Protocol

**Participants**: 8 signers
- 3 Deaf/hard-of-hearing native or fluent ASL signers
- 3 hearing learners (intermediate ASL)
- 2 hearing learners (beginner ASL)
This mix tests signer-independence and informs enrollment value.

**Per sign, per signer**:

| Condition | Variations |
|-----------|-----------|
| Repetitions | 5 |
| Background | (1) plain white wall, (2) cluttered desk, (3) outdoor/window |
| Lighting | (1) bright daylight, (2) dim indoor |
| Hand dominance | Left-handed and right-handed variants collected |
| Signing speed | (1) normal, (2) slow/deliberate, (3) fast/natural |

**Isolated clips**: 8 signers × 5 reps × 3 backgrounds × 2 lighting ≈ 240 clips per sign × 150 signs = **36,000 isolated clips**

**Continuous sentence clips**: Each signer records 25 sentence prompts (3–7 signs each) across 3 backgrounds = 8 × 25 × 3 = **600 sentence clips**

**Total estimated**: ~36,600 clips. Realistically achievable in 4–6 weeks by a 3-person team.

---

### Data Collection Setup

**Hardware**:
- HP Snapdragon PC (target device) — built-in camera, same as deployment
- 720p or 1080p webcam
- No depth sensor required (RGB only, matching deployment)

**Software**:
- OpenCV capture script (provided in `data/collect.py`)
- Automatic landmark extraction during collection (quality check on-the-fly)
- Metadata JSON per clip: signer_id, sign_label, speed, background, lighting, handedness

**Quality criteria per clip**:
- Both hands visible in > 80% of frames
- No frame drops > 5 consecutive frames
- Lighting quality score > 0.3 (see `camera.check_lighting_quality()`)

---

### Pre-processing Pipeline

```
Raw video clip (.mp4)
  ↓
Frame extraction @ 30 FPS
  ↓
MediaPipe Holistic landmark extraction per frame
  ↓
Zero-mean normalization + scale normalization (shoulder-width)
  ↓
Temporal padding/trimming to 64 frames
  ↓
Save: landmarks.npy [64, 225] + label.json
  ↓
Train/val/test split: 70/15/15 (stratified by signer)
```

**Important**: Test set always contains at least 2 unseen signers (signer-independent evaluation).

---

### Data Augmentation

To increase effective dataset size and robustness:

| Augmentation | Implementation | Rationale |
|-------------|----------------|-----------|
| Landmark jitter | Add Gaussian noise σ=0.01 to landmark coordinates | Robustness to tracking noise |
| Horizontal flip | Mirror x-coordinates (swap left/right hand) | Left-handed signer simulation |
| Speed perturbation | Interpolate/skip frames ±30% | Handles fast/slow signers |
| Rotation | Rotate 2D projection ±15° | Camera angle variation |
| Temporal shift | Shift sequence start ±5 frames | Segmentation robustness |

Each clip produces 5 augmented variants → effective dataset: ~180,000 isolated clips.

---

### Train / Val / Test Split

```
WLASL (pre-training backbone):
  Train: 80%
  Val:   10%
  Test:  10%

Custom 150-sign dataset:
  Train: 70% (6 signers)
  Val:   15% (1 signer, seen during training)
  Test:  15% (2 signers, UNSEEN — signer-independent test)
```

The signer-independent test set is the most important metric: it reveals how much the base model generalizes before personalization.

---

### Baseline vs. Personalized Comparison

The evaluation explicitly measures:

1. **Baseline accuracy**: test set accuracy with no enrollment (signer-independent)
2. **Enrolled accuracy**: same user, same signs, after 5-minute enrollment
3. **Delta**: the improvement from personalization

A meaningful result would show ≥ +10% accuracy improvement post-enrollment for the enrolled user, with no regression for other users.
