# Final release audit

Audit date: 2026-11-19. This report records the final local quality and publication audit for the Smart Dice Board Game RL course project.

## Scope inspected

Inspected the assignment DOCX, environment and board generation, tabular and SB3 agents, baseline and supplemental training/evaluation scripts, experiment configuration, all result CSV groups, model inventories, audit outputs, README and requirements, all report sources, and the rendered final report. The supplied assignment DOCX and the Codex project instruction file are excluded from publication; their local copies remain unchanged.

## Assignment compliance

The source assignment says the agent chooses a die value from 1–6 in the core task and calls random dice an optional stochastic challenge. The current core environment is therefore correctly deterministic and controllable. The assignment requires an environment, but does not name Gymnasium or Stable-Baselines3. The project uses Gymnasium and SB3 as implementation choices, and exceeds the minimum of three approaches with Q-Learning, SARSA, DQN, PPO, and A2C. Overshoot is documented as a stay-in-place transition with −10 reward that consumes a move.

See [assignment compliance](assignment_compliance.md) for the row-by-row evidence, verification, status, and unimplemented optional challenges.

## Report audit

- The final Markdown report references 20 figures; all 20 image paths exist.
- The final DOCX embeds 20 figures; the PDF contains all 20 captions in order and renders to 19 pages.
- Every rendered page was visually inspected. The standalone cover page, tables, captions, charts, margins, and section flow were checked.
- The gamma table's 10/10 Wilson interval [72.2%, 100%] is interpreted as uncertainty across ten independent training seeds on that tested board/configuration, not as a guarantee across arbitrary boards. Deterministic repeated rollouts are not counted as independent policy training samples.
- Temporary checkpoint threshold crossings are distinguished from sustained convergence and final evaluation. The 100-square SARSA baseline's episode-1,500 crossing is explicitly described alongside its 0% final success.
- The report separates original baseline, longer-budget, gamma, reward, generalization, tuning, and artifact-integrity studies. Results, failed runs, changed reward/observation conditions, and weak generalization are disclosed.

## Reproduced results

| Check | Verified result |
|---|---|
| Original policy integrity | 45/45 valid |
| Earlier improved-policy integrity | 27/27 valid |
| Original evaluation reconciliation | 48 rows; zero metric mismatches |
| Original grouped algorithm aggregates | Matches raw rows within 5e-13 |
| Independent BFS | 3, 3, and 9 moves on the three saved boards |
| 10,000-episode random audit | 9,999/10,000 wins on the 100-square board; one 400-step failure |
| Complete SB3 artifact reload audit | 96/96 valid actions |
| New tabular artifact audit | 264 unique archives with valid shape and finite values |
| Held-out generalization reload | 378/378 saved per-board metric rows reproduced |
| DRL tuning reload | 60/60 saved per-board metric rows reproduced |
| CSV readability | 38/38 files parsed |

The 10-seed gamma study solved all ten Q-Learning runs at γ=.80/.90 and all ten SARSA runs at each tested γ on the 100-square fixed board. Q-Learning at γ=.95/.99 had 0/10 final wins. Successful routes averaged 10–10.1 moves, compared with BFS's 9. This is a fixed-layout result, not a general board guarantee.

The reward-variant study's goal-focused Q-Learning reached 100% success and 9 average moves in its three tested runs. For held-out 100-square DRL generalization, the best multi-board result was DQN at 11.1%; fixed-board DRL had no test wins. The DRL tuning study's best mean result was balanced A2C at 33.3% across its two training seeds and three test layouts. None supports a claim of reliable DRL transfer.

## Tests and commands checked

- `pytest -q`: **16 passed**.
- Python syntax compilation: **27 project and report modules passed**.
- All five documented learned-policy play commands were run; each model loaded and reached the goal. The command also showed random play before the learned policy.
- All three README image links and all listed experiment module paths were checked; no missing references were found.
- `pip check` reported conflicts among unrelated packages already present in the shared interpreter (`gradio`/`gradio-client`, `langchain-huggingface`, and `pyannote-audio` constraints on NumPy, pandas, websockets, or PyTorch). The project imports, environment checkers, and full tests pass; no global packages were changed. A clean isolated install was not run because the environment already contains the full project stack and a separate PyTorch install would be heavyweight.
- The root CLI does not have a per-algorithm training flag; its `train` command trains all five algorithms. README does not invent single-algorithm training options.

## Release selection and exclusions

The public source release retains the complete source package, tests, board/config files, result CSVs, experiment figures, audit evidence, requirements, and Markdown/Word/PDF reports. Five seed-17 30-square policies are included for the Q-Learning, SARSA, DQN, PPO, and A2C demos. Other original and supplemental model archives remain in the local workspace and are excluded by `.gitignore` to keep the public release small; model generation commands and saved result evidence are documented.

Excluded from publication: the supplied assignment DOCX (distribution rights not established), the Codex project brief, Python/pytest caches, report render caches, duplicate historical report renders, local project inventory, and non-demo model/checkpoint archives. No credential, personal email, or local machine path was found in the text files selected for release. No license was added because the project owner has not selected a license; see release limitations below.

## GitHub publication

- Target: [omar320230147-byte/smart-dice-board-game-rl](https://github.com/omar320230147-byte/smart-dice-board-game-rl)
- Uploads were made through the authenticated Google Chrome session and committed directly to `main`; no terminal Git credentials were changed.
- The public repository now contains the baseline comparison graphs (`results/plots`, 6 PNGs), original baseline result tables (`results/csv`, 5 files), independent audit evidence (`results/audit`, 8 files), and the full ten-seed gamma sweep bundle (`results/improved/gamma_sweep_all_boards`, 5 files).
- Latest verified publication at the time of this audit: `4580f25ab87f80d82682168a69dabbe0ce13e4a3` (“Publish 100-square gamma sweep results”).
- The compact release includes five seed-17, 30-square demo policies. Other research model archives and checkpoints are retained in the local workspace and excluded from the public release.
- The root README renders the baseline comparison and ten-seed gamma figures. The held-out generalization figure and other supplemental files remain local; the README preview was removed until that image is published, while the measured generalization result remains stated with its limitation.
- README and release notes are being corrected to match the curated public tree. This audit does not alter or replace any experimental result.

## Remaining limitations

DRL transfer to unseen 100-square layouts remained weak at the tested CPU budgets. The gamma study varies training seeds on fixed layouts and does not establish performance over a distribution of boards. The random baseline is unusually effective on the one saved 100-square board. Optional random-dice, one-time snake-avoidance, and multiplayer challenges were not implemented. The assignment file is not redistributed; a license is not assigned in this release. GitHub publication remains incomplete as described above.

