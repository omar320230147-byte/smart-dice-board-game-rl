# Software versions used for the checked run

These versions were queried from the active project interpreter on 2026-11-19. `requirements.txt` declares the supported minimum/range constraints; it does not lock every transitive dependency.

| Component | Observed version | Requirement |
|---|---:|---|
| Python | 3.12.2 | ≥3.10 |
| Gymnasium | 1.1.1 | ==1.1.1 |
| Stable-Baselines3 | 2.6.0 | ==2.6.0 |
| PyTorch | 2.5.1+cu121 | ≥2.3, <3.0 |
| NumPy | 2.4.5 | ≥1.24 |
| pandas | 3.0.3 | ≥2.0 |
| matplotlib | 3.10.9 | ≥3.7 |
| python-docx | 1.2.0 | ≥1.1 |
| pytest | 8.4.2 | ≥8.0 |

Although the installed PyTorch wheel contains CUDA libraries, experiment constructors explicitly select CPU execution. A CPU-only wheel can be substituted if preferred; the project does not require a GPU.
