# Validation Harness

```powershell
python .\experiments\validation\compare_games.py --case pointing_game_shapley --input assets\dog_and_hydrant.jpg --text "dog" --random-state 0 --num-coalitions 4 --batch-size 2 --tolerance 1e-4 --cuda
```

Runs an old-vs-new Game pipeline comparison for the selected image/text input and writes the CSV result to `experiments/validation/results/`.

Use the migrated Python file name without `.py` for `--case`.
