"""
Train Temporal Encoder + Translation Decoder
=============================================
Step 1: Pre-train on WLASL (2,000 signs, CC-BY)
Step 2: Fine-tune on custom 150-sign dataset

Run on a machine with a GPU. Produces:
  - temporal_encoder.pt       (PyTorch checkpoint)
  - translation_decoder.pt
  - personalization_mlp.pt

Then export with: python training/export_onnx.py

Usage:
    python training/train_encoder.py --stage pretrain --epochs 50
    python training/train_encoder.py --stage finetune --epochs 30 --checkpoint checkpoints/pretrain_best.pt
"""

import argparse
import logging
import json
from pathlib import Path

import numpy as np

log = logging.getLogger("signbridge.training.encoder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
CKPT_DIR   = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
INPUT_DIM   = 225    # Landmark feature dimension
D_MODEL     = 256    # Transformer model dimension
N_HEADS     = 8
N_ENC_LAYERS = 4
N_DEC_LAYERS = 2
SEQ_LEN     = 64
DROPOUT     = 0.1
BATCH_SIZE  = 32
LR          = 3e-4
WEIGHT_DECAY = 1e-4


def build_encoder():
    """Build the temporal encoder model (PyTorch)."""
    try:
        import torch
        import torch.nn as nn

        class TemporalEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj  = nn.Linear(INPUT_DIM, D_MODEL)
                self.pos_embed   = nn.Embedding(SEQ_LEN, D_MODEL)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=D_MODEL, nhead=N_HEADS,
                    dim_feedforward=1024, dropout=DROPOUT,
                    batch_first=True, activation="gelu",
                )
                self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_ENC_LAYERS)
                self.pool        = nn.AdaptiveAvgPool1d(1)
                self.norm        = nn.LayerNorm(D_MODEL)

            def forward(self, x):                         # x: [B, T, INPUT_DIM]
                x  = self.input_proj(x)                  # [B, T, D_MODEL]
                t  = torch.arange(x.size(1), device=x.device)
                x  = x + self.pos_embed(t).unsqueeze(0)
                x  = self.transformer(x)                 # [B, T, D_MODEL]
                x  = self.norm(x)
                x  = self.pool(x.transpose(1, 2)).squeeze(-1)  # [B, D_MODEL]
                return x

        return TemporalEncoder()

    except ImportError:
        log.error("PyTorch not installed. Training requires: pip install torch torchvision")
        return None


def build_decoder(vocab_size: int):
    """Build the translation decoder (PyTorch)."""
    try:
        import torch
        import torch.nn as nn

        class TranslationDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj   = nn.Linear(D_MODEL, D_MODEL)
                dec_layer = nn.TransformerDecoderLayer(
                    d_model=D_MODEL, nhead=N_HEADS,
                    dim_feedforward=1024, dropout=DROPOUT,
                    batch_first=True, activation="gelu",
                )
                self.transformer = nn.TransformerDecoder(dec_layer, num_layers=N_DEC_LAYERS)
                self.head        = nn.Linear(D_MODEL, vocab_size)

            def forward(self, memory, tgt):  # memory: [B, D_MODEL], tgt: [B, T_tgt, D_MODEL]
                memory = self.proj(memory).unsqueeze(1)   # [B, 1, D_MODEL]
                out    = self.transformer(tgt, memory)    # [B, T_tgt, D_MODEL]
                return self.head(out)                     # [B, T_tgt, vocab_size]

        return TranslationDecoder()

    except ImportError:
        return None


def build_personalization_mlp():
    """Build the personalization MLP (PyTorch)."""
    try:
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(D_MODEL + 64, 512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, D_MODEL),
        )
    except ImportError:
        return None


def load_dataset(split: str, data_dir: Path) -> list:
    """
    Load pre-processed landmark sequences.
    Expected format: data/processed/{split}/
        each sample: {id}.npy  (float32 [64, 225]) + {id}.json ({"label": "HELP"})
    """
    processed_dir = data_dir / "processed" / split
    if not processed_dir.exists():
        log.warning(f"Processed data not found at {processed_dir}")
        log.warning("Run: python data/preprocessing.py first")
        return []

    samples = []
    for npy_path in sorted(processed_dir.glob("*.npy")):
        label_path = npy_path.with_suffix(".json")
        if not label_path.exists():
            continue
        seq   = np.load(str(npy_path))
        label = json.loads(label_path.read_text())["label"]
        samples.append((seq, label))

    log.info(f"Loaded {len(samples)} samples for split '{split}'")
    return samples


def train(args):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        log.error("PyTorch required for training. Install: pip install torch")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training device: {device}")

    # Load vocabulary
    vocab_path = DATA_DIR / "vocabulary.json"
    vocab_data  = json.loads(vocab_path.read_text())
    vocab       = vocab_data.get("token_to_id", {})
    vocab_size  = max(vocab.values()) + 1 if vocab else 100
    log.info(f"Vocabulary size: {vocab_size}")

    # Build models
    encoder = build_encoder().to(device)
    decoder = build_decoder(vocab_size).to(device)
    perso   = build_personalization_mlp().to(device)

    if encoder is None:
        return

    # Load data
    train_data = load_dataset("train", DATA_DIR)
    val_data   = load_dataset("val",   DATA_DIR)

    if not train_data:
        log.warning("No training data found. Skipping training loop.")
        log.warning("After collecting data, run: python data/preprocessing.py")
        # Save untrained model as placeholder
        torch.save(encoder.state_dict(), CKPT_DIR / "temporal_encoder_init.pt")
        log.info(f"Saved initial (untrained) encoder to {CKPT_DIR}")
        return

    # Optimizer
    all_params = list(encoder.parameters()) + list(decoder.parameters()) + list(perso.parameters())
    optimizer  = optim.AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion  = nn.CrossEntropyLoss(ignore_index=0)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        encoder.train(); decoder.train(); perso.train()
        total_loss = 0.0
        n_batches  = 0

        # Mini-batch loop (simplified — no DataLoader for brevity)
        for i in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[i: i + BATCH_SIZE]
            seqs  = torch.tensor(
                np.stack([s for s, _ in batch]), dtype=torch.float32
            ).to(device)                             # [B, 64, 225]

            labels = torch.tensor(
                [vocab.get(l, 3) for _, l in batch], dtype=torch.long
            ).to(device)                             # [B]

            optimizer.zero_grad()
            emb    = encoder(seqs)                   # [B, 256]
            # Simple classification head for encoder pre-training
            logits = decoder.head(emb.unsqueeze(1)).squeeze(1)  # [B, vocab_size]
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        log.info(f"Epoch {epoch:3d}/{args.epochs}  train_loss={avg_loss:.4f}")

        # Validation
        if val_data and epoch % 5 == 0:
            encoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in range(0, len(val_data), BATCH_SIZE):
                    batch  = val_data[i: i + BATCH_SIZE]
                    seqs   = torch.tensor(np.stack([s for s, _ in batch]), dtype=torch.float32).to(device)
                    labels = torch.tensor([vocab.get(l, 3) for _, l in batch], dtype=torch.long).to(device)
                    emb    = encoder(seqs)
                    logits = decoder.head(emb.unsqueeze(1)).squeeze(1)
                    val_loss += criterion(logits, labels).item()
            avg_val = val_loss / max(len(val_data) // BATCH_SIZE, 1)
            log.info(f"  val_loss={avg_val:.4f}")
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(encoder.state_dict(), CKPT_DIR / "temporal_encoder_best.pt")
                torch.save(decoder.state_dict(), CKPT_DIR / "translation_decoder_best.pt")
                torch.save(perso.state_dict(),   CKPT_DIR / "personalization_mlp_best.pt")
                log.info(f"  ✅  New best saved (val_loss={avg_val:.4f})")

    log.info("Training complete.")
    log.info(f"Best checkpoints saved to {CKPT_DIR}")
    log.info("Next step: python training/export_onnx.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SignBridge temporal encoder")
    parser.add_argument("--stage",      choices=["pretrain", "finetune"], default="finetune")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    train(args)
