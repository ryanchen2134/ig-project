"""train2_clean.py – contrastive training with MoCo‑style queue
----------------------------------------------------------------
Implements the *once‑on‑device* queue allocation so every tensor lives on the
same device (CPU or CUDA) and concatenation never triggers a device mismatch.
The rest of the script is identical to your original *train2.py* except where
marked ############### CHANGED ###############.
"""
import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.image_backbone import build_resnet18  # <- your existing imports
from models.decoder.contrastive import NTXentLoss
from dataset import load_dataset  # <- whatever returns (train_ds, val_ds, test_ds)

# ──────────────────────────── helper utils ────────────────────────────

def init_queue(queue_size: int, embedding_dim: int, device: torch.device):
    """Allocate an empty FIFO queue *on the same device as the model*."""
    return torch.zeros(queue_size, embedding_dim, device=device)


def enqueue_dequeue(queue: torch.Tensor, new_keys: torch.Tensor, ptr: int):
    """MoCo‑style enqueue and dequeue (update in‑place).

    Args
    -----
    queue:   [Q, D] FIFO buffer on *device*
    new_keys:[B, D] mini‑batch embeddings on same device
    ptr:     current position in the circular buffer
    Returns
    -------
    ptr: new pointer after enqueuing
    """
    Q = queue.size(0)
    batch_size = new_keys.size(0)

    assert batch_size <= Q, "queue size must be ≥ batch size"

    queue[ptr:ptr + batch_size] = new_keys.detach()
    ptr = (ptr + batch_size) % Q
    return ptr

# ────────────────────────────── training ──────────────────────────────

def train(model, loss_fn, optimiser, train_loader, queue_img, queue_song, queue_ptr,
          device, grad_accum_steps, epochs):
    history = []
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimiser.zero_grad(set_to_none=True)

        for step, (img, song) in enumerate(train_loader):
            img = img.to(device, non_blocking=True)
            song = song.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                img_emb, song_emb = model(img, song)

                # ############### CHANGED ###############
                # Concatenate with negatives already on *device*
                img_cat = torch.cat([img_emb, queue_img], dim=0)  # [B+Q, D]
                song_cat = torch.cat([song_emb, queue_song], dim=0)
                # ############### END CHANGE ###############

                loss = loss_fn(img_emb, song_emb, model_sim=(img_cat @ song_cat.T))
                loss = loss / grad_accum_steps  # gradient accumulation

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

            # update queues after *every physical batch* (MoCo v2 style)
            queue_ptr = enqueue_dequeue(queue_img, img_emb, queue_ptr)
            queue_ptr = enqueue_dequeue(queue_song, song_emb, queue_ptr)

        history.append(running_loss / len(train_loader))
        print(f"Epoch {epoch:02d} – loss {history[-1]:.4f}")
    return history

# ──────────────────────────────── main ────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="# of steps to accumulate gradients (simulated batch_size = batch * steps)")
    parser.add_argument("--queue_size", type=int, default=1024,
                        help="MoCo memory queue length (0 disables queue)")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=15)
    args = parser.parse_args()

    # dataset
    train_ds, val_ds, test_ds = load_dataset()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # model
    backbone = build_resnet18(pretrained=True)
    model = nn.DataParallel(backbone).to(device)

    loss_fn = NTXentLoss(temperature=0.07, hard_negative_weight=0.2)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ############### CHANGED ###############
    embedding_dim = loss_fn.embedding_dim  # assume NTXentLoss exposes this
    queue_img = init_queue(args.queue_size, embedding_dim, device)
    queue_song = init_queue(args.queue_size, embedding_dim, device)
    queue_ptr = 0  # ring buffer pointer
    # ############### END CHANGE ###############

    history = train(model, loss_fn, optimiser, train_loader,
                    queue_img, queue_song, queue_ptr,
                    device, args.grad_accum_steps, args.epochs)

    print("Training complete. Final loss:", history[-1])

if __name__ == "__main__":
    main()
