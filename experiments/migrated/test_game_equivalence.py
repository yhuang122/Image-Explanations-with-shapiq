# tests/test_game_equivalence.py
"""
Equivalence test: src.game_huggingface vs Game/game_huggingface (via factory).

Run all models:
    pytest experiments/migrated/test_game_equivalence.py -v

Run a specific model:
    pytest experiments/migrated/test_game_equivalence.py -v -k "clip-vit-base-patch32"

Run with extra coalitions (slow):
    pytest experiments/migrated/test_game_equivalence.py -v --n-coalitions 200
"""
import sys

from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ImputerFactory import ImageImputerFactory
from Game import VisionLanguageGame as NewGame
from src.game_huggingface import VisionLanguageGame as OldGame


# ── CLI options ────────────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption("--n-coalitions", type=int, default=50)


@pytest.fixture(scope="session")
def n_coalitions(request):
    return request.config.getoption("--n-coalitions")


# ── parametrize over models ────────────────────────────────────────────────────

MODELS = [
    "openai/clip-vit-base-patch32",
    "google/siglip-base-patch16-224",
    "google/siglip2-base-patch16-224",
]


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", params=MODELS)
def model_name(request):
    return request.param


@pytest.fixture(scope="module")
def device():
    return 0 if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def model_and_processor(model_name, device):
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


@pytest.fixture(scope="module")
def sample_inputs():
    """Minimal deterministic inputs — same across all model tests."""
    rng = np.random.default_rng(seed=0)
    image = Image.fromarray(
        rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    text = "a cat sitting on a mat"
    return image, text


@pytest.fixture(scope="module")
def games(model_and_processor, sample_inputs):
    model, processor = model_and_processor
    image, text = sample_inputs

    imputer = ImageImputerFactory().build(model, processor, image, text)
    new_game = NewGame(imputer, batch_size=16)
    old_game = OldGame(model, processor, image, text, batch_size=16)

    return new_game, old_game, imputer


# ── tests ──────────────────────────────────────────────────────────────────────

TOL = 1e-4


def test_n_players_match(games):
    new_game, old_game, _ = games
    assert new_game.n_players_image == old_game.n_players_image, (
        f"n_players_image mismatch: {new_game.n_players_image} vs {old_game.n_players_image}"
    )
    assert new_game.n_players_text == old_game.n_players_text, (
        f"n_players_text mismatch: {new_game.n_players_text} vs {old_game.n_players_text}"
    )


def test_empty_value(games):
    new_game, old_game, _ = games
    assert abs(new_game.empty_value - old_game.empty_value) < TOL, (
        f"empty_value: new={new_game.empty_value:.6f}  old={old_game.empty_value:.6f}"
    )


def test_full_value(games):
    new_game, old_game, _ = games
    assert abs(new_game.full_value - old_game.full_value) < TOL, (
        f"full_value: new={new_game.full_value:.6f}  old={old_game.full_value:.6f}"
    )


def test_value_function_equivalence(games, n_coalitions=20):
    new_game, old_game, _ = games
    n_players = new_game.n_players_image + new_game.n_players_text

    rng = np.random.default_rng(seed=0)
    coalitions = rng.integers(0, 2, (n_coalitions, n_players), dtype=bool)
    coalitions[0] = False   # all-zeros
    coalitions[1] = True    # all-ones

    out_new = new_game.value_function(coalitions)
    out_old = old_game.value_function(coalitions)
    abs_diff = np.abs(out_new - out_old)

    worst = int(np.argmax(abs_diff))
    assert abs_diff.max() < TOL, (
        f"Max diff {abs_diff.max():.6f} exceeds {TOL}\n"
        f"  worst coalition [{worst}]: "
        f"new={out_new[worst]:.6f}  old={out_old[worst]:.6f}\n"
        f"  mean diff: {abs_diff.mean():.6f}"
    )


def test_model_type_detected(games):
    """Smoke-test that the factory correctly detected the model type."""
    _, _, imputer = games
    assert imputer.config.model_type in {"clip", "siglip", "siglip2"}, (
        f"Unexpected model_type: {imputer.config.model_type!r}"
    )