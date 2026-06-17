"""
Quick equivalence check: src.game_huggingface vs Game/game_huggingface (via factory).
Works for CLIP, SigLIP, and SigLIP-2.

Run on the GPU machine:
    python experiments/migrated/compare_games.py --model_name openai/clip-vit-base-patch32
    python experiments/migrated/compare_games.py --model_name google/siglip-base-patch16-224
    python experiments/migrated/compare_games.py --model_name google/siglip2-base-patch16-224
"""
import argparse
import sys
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
parser.add_argument("--n_coalitions", type=int, default=50)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image

import src
from shapiq.imputer.vision import VisionImputerFactory
from shapiq.imputer.vision import VisionLanguageGame as NewGame
from src.game_huggingface import VisionLanguageGame as OldGame

np.random.seed(args.seed)

# ── load model (works for CLIP, SigLIP, SigLIP-2) ────────────────────────────
device = 0 if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(args.model_name).to(device)
processor = AutoProcessor.from_pretrained(args.model_name)

# ── minimal sample: a solid-color image + short text ─────────────────────────
image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
text = "a cat sitting on a mat"

# ── build both games ──────────────────────────────────────────────────────────
factory = VisionImputerFactory()
imputer = factory.build(model, processor, image, text)
new_game = NewGame(imputer, batch_size=16)

old_game = OldGame(model, processor, image, text, batch_size=16)

assert new_game.n_players_image == old_game.n_players_image, "n_players_image mismatch"
assert new_game.n_players_text == old_game.n_players_text, "n_players_text mismatch"
n_players = new_game.n_players_image + new_game.n_players_text
print(f"model_type detected: {imputer.config.model_type}")
print(f"n_players_image={new_game.n_players_image}, n_players_text={new_game.n_players_text}")

# ── random coalitions ─────────────────────────────────────────────────────────
coalitions = np.random.randint(0, 2, (args.n_coalitions, n_players), dtype=bool)
# always include all-zeros and all-ones
coalitions[0] = False
coalitions[1] = True

# ── compare ───────────────────────────────────────────────────────────────────
out_new = new_game.value_function(coalitions)
out_old = old_game.value_function(coalitions)

abs_diff = np.abs(out_new - out_old)
print(f"max absolute diff : {abs_diff.max():.6f}")
print(f"mean absolute diff: {abs_diff.mean():.6f}")
print(f"empty_value  new={new_game.empty_value:.6f}  old={old_game.empty_value:.6f}")
print(f"full_value   new={new_game.full_value:.6f}  old={old_game.full_value:.6f}")

tol = 1e-4
if abs_diff.max() < tol:
    print(f"\nPASS — outputs match within {tol}")
else:
    print(f"\nFAIL — max diff {abs_diff.max():.6f} exceeds {tol}")
    worst = np.argmax(abs_diff)
    print(f"  worst coalition index: {worst}")
    print(f"  new={out_new[worst]:.6f}  old={out_old[worst]:.6f}")
