# Conformal Prediction Experiments (Reproducible Pipeline)

This README provides **complete commands** to reproduce all conformal-prediction experiments, figures, and tables from scratch.

## 1) Environment requirements

- Python 3.10+ (tested on Python 3.12)
- Allowed libraries only:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

## 2) Install dependencies

From repository root:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn matplotlib
```

## 3) (Optional) Verify project files exist

```bash
python - <<'PY'
from pathlib import Path
required = [
    'data.py', 'models.py', 'conformal.py', 'experiments.py',
    'plots.py', 'tables.py', 'main.py'
]
missing = [p for p in required if not Path(p).exists()]
print('Missing:', missing if missing else 'None')
PY
```

## 4) Run the full experiment pipeline

Single command (recommended):

```bash
python main.py
```

This command will automatically:
1. load/download datasets,
2. preprocess features/labels,
3. run all experiments across seeds,
4. save raw and aggregated results,
5. generate all figures,
6. generate all tables.

## 5) Quick sanity checks

### 5.1 Syntax check

```bash
python -m compileall data.py models.py conformal.py experiments.py plots.py tables.py main.py
```

### 5.2 Output existence check

```bash
python - <<'PY'
from pathlib import Path
checks = [Path('results'), Path('figures'), Path('tables')]
for p in checks:
    print(f"{p}:", 'OK' if p.exists() else 'MISSING')
PY
```

## 6) Clean generated artifacts

If you want to remove generated results and rerun cleanly:

```bash
rm -rf results figures tables
```

## 7) Re-run from a clean state

```bash
rm -rf results figures tables
python main.py
```

---

## Notes

- The code tries to load Adult Income from OpenML. If network/OpenML is unavailable, it uses a fallback public dataset so the pipeline remains runnable.
- Randomness is controlled via fixed seeds in `main.py` (`[0, 1, 2, 3, 4]`).
