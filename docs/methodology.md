# Methodology and experiment map

## Core task

The controllable-dice game is a deterministic episodic Markov decision process. A state is the current square `s ∈ {1,…,N}`; action `a ∈ {0,…,5}` corresponds to the selected die value `a+1`. The player advances unless the target exceeds the goal, in which case it remains in place. Landing on a snake or ladder applies one immediate map transition. The goal terminates; the episode truncates after `4N` moves. Core reward is +100 on goal, +20 on ladder, −1 on a normal landing, −20 on a snake, and −10 on overshoot.

The generated fixed layouts use seeds 303, 505, and 1008 for the 30-, 50-, and 100-square boards. Their BFS shortest paths are 3, 3, and 9 moves. BFS acts as a planning lower bound with complete board knowledge; the RL agents use the same legal six actions and transition rules.

## Original baseline

The original configuration is `config/experiments.json`: Q-Learning and SARSA train for 3,500 episodes; DQN, PPO, and A2C train for 6,000 environment steps. The three training seeds are 17, 29, and 41. Final evaluation uses 100 episodes per policy in a separate environment. Training progress/checkpoints and final policy evaluation are distinct. A deterministic policy on the fixed board produces the same path for every rollout.

Tabular defaults use α=.15, γ=.99, and ε starting at 1.0, decaying by .999 per episode to .03. SB3 uses CPU MLP policies (64,64), with algorithm settings recorded in `report/final_report.md` and source/config files. The baseline suite is executed by `python -m smart_dice_rl train` and regenerates files under `models/` and `results/csv/`.

## Supplemental studies

| Study | Training setup | Distinction from original baseline | Data |
|---|---|---|---|
| Longer budget | New seeds; 20,000 tabular episodes or 25,000 SB3 timesteps on the fixed 100-square board | More interactions; not pooled into original result table | `results/improved/csv/` |
| Gamma sweep | Ten independent seeds per algorithm and γ∈{.80,.90,.95,.99}; 5,000 episodes; unchanged core reward | Discount factor and seed set differ; extended to all three fixed boards | `results/improved/gamma_sweep*/` |
| Reward variants | Q-Learning/SARSA, three seeds, 5,000 episodes per condition | Reward function changes; original reward data are retained as a separate condition | `results/improved/reward_experiments/` |
| Generalization | Board-aware 202-feature observation; nine training boards, three validation boards, nine test boards across sizes; 15,000 SB3 steps, three training seeds per regime | Observation space and board distribution differ; test board seeds are disjoint | `results/improved/generalization/` |
| DRL tuning | Two settings per SB3 algorithm; two seeds, 10,000 steps, one validation board and three held-out boards | Hyperparameters vary; not pooled into original SB3 outcomes | `results/improved/drl_experiments/` |

The reward study tests an original reward, a reduced ladder bonus, a goal-focused reward, and potential shaping. Shaped rewards change the optimization objective. Potential shaping uses `F(s,s')=γΦ(s')−Φ(s)` with normalized position potential and zero goal potential. Because the environment has an artificial time-limit truncation, strict finite-horizon invariance requires remaining time in the state; the empirical result is not a universal policy-invariance guarantee.

## Metrics and inference

Success is the fraction reaching the goal. Moves include all environment transitions; failures are counted at the configured step limit. Cumulative reward is the undiscounted sum of environment rewards. Training time is wall-clock time. Convergence is reported at the third consecutive checkpoint with at least 95% validation success; this first-crossing metric is not proof that the final policy remains successful. The original SARSA 100-square runs demonstrate this distinction.

The 10/10 gamma results use Wilson 95% intervals over independent training seeds. They characterize the fraction of the tested training seeds solving a fixed board under that run configuration; they do not guarantee all boards will be solved. Repeated rollouts of a deterministic saved policy are not independent training samples.

## Reproduction commands

Run from the repository root:

```bash
pytest -q
python -m smart_dice_rl train
python -m smart_dice_rl evaluate
python -m smart_dice_rl.final_audit
python -m smart_dice_rl.finalize_outputs
python -m smart_dice_rl.gamma_sweep
python -m smart_dice_rl.gamma_sweep_all_boards
python -m smart_dice_rl.reward_experiments
python -m smart_dice_rl.run_generalization
python -m smart_dice_rl.drl_tuning
```

The baseline `train` writes the baseline file paths and can replace baseline outputs. Supplemental experiments write to their own subdirectories and leave baseline data in place.
