# Migrated Experiments

This folder contains migrated copies of selected experiment scripts.
The original scripts in `experiments/` are kept unchanged.

## A1.5

`pointing_game_shapley.py` is copied from `experiments/pointing_game_shapley.py`
and migrated from the old `src.game_huggingface.VisionLanguageGame` constructor
to the new `ImageImputerFactory` + `Game.VisionLanguageGame` pipeline.

## A1.6

`pointing_game_crossmodal.py` is copied from
`experiments/pointing_game_crossmodal.py` and migrated to the same new pipeline.
The crossmodal visualization reads `model_type` from the generated imputer.
