# Validation Harness

```powershell
python .\experiments\validation\compare_games.py --case pointing_game_shapley --input assets\dog_and_hydrant.jpg --text "dog" --random-state 0 --num-coalitions 4 --batch-size 2 --tolerance 1e-4 --cuda
```

Runs an old-vs-new Game pipeline comparison for the selected image/text input and writes the CSV result to `experiments/validation/results/`.

Use the migrated Python file name without `.py` for `--case`.

```powershell
python .\experiments\validation\summarize_results.py
```

Reads comparison CSV files from `experiments/validation/results/` and writes `experiments/validation/results/summary.csv`.
It also writes `experiments/validation/results/max_abs_diff_summary.png`.

Benchmark note: future benchmark runs should validate the complete pipeline end-to-end, not only the value-function equivalence harness.
