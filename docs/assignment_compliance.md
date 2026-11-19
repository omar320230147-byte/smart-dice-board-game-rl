# Assignment compliance

The source assignment `Smart_Dice_Board_Game_RL_Course_Project.docx` was read directly from the project root. It is intentionally not included in the public release. This matrix summarizes its requirements and checks them against the implementation and experiment files.

| Assignment requirement | Evidence in the project | Verification method | Status | Remaining action |
|---|---|---|---|---|
| Core game is a board of N squares; start at 1 and reach N | `smart_dice_rl/environment.py` | Read assignment text and environment reset/termination behavior; environment tests | Pass | None |
| In the core game, the agent chooses a die outcome from 1–6 | Six discrete actions in `SmartDiceEnv`; action index `a` means die value `a+1` | Assignment explicitly says controllable die; source and test inspection | Pass; deterministic action semantics are correct | Optional random-dice extension is separate and not substituted into baseline results |
| Apply snake/ladder immediately after landing | Transition mapping in `SmartDiceEnv` | Read assignment and inspect `step`; unit tests cover transitions | Pass | None |
| Define invalid move / overshoot behavior consistently | Overshoot stays in place, receives −10, and consumes a move | Assignment permits a documented invalid-action rule; checked report and environment | Pass; implementation choice is documented | None |
| Define state and action spaces | State is current square; six legal action values | Inspect Gymnasium spaces and MDP section in report | Pass | None |
| Define rewards for goal, ladder, ordinary square, snake, and invalid move | +100, +20, −1, −20, and −10 respectively | Read assignment reward table and compare to environment/reward tests | Pass | Alternative reward functions are explicitly separate studies |
| Implement RL and epsilon-greedy exploration | Q-Learning and SARSA tabular agents | Source inspection and test suite | Pass | None |
| Compare at least three RL approaches | Q-Learning, SARSA, DQN, PPO, A2C | Original result CSVs, models, report | Pass; five algorithms are implemented and trained | None |
| Environment API / framework | The assignment requires an environment but does not name Gymnasium | Source assignment read; `SmartDiceEnv` and Gymnasium API tests inspected | Implemented with Gymnasium, beyond the stated framework requirement | None |
| Stable-Baselines3 for DRL | The assignment does not require SB3; it names DQN as an advanced challenge | Source assignment read; DQN/PPO/A2C imports, training code, and archives inspected | Implemented with SB3, beyond the stated requirement | None |
| Run all three required board configurations | `config/experiments.json` and `data/generated_boards/` | Compare assignment table to board JSON and CSVs | Pass: 30/3/3, 50/5/5, 100/8/8 | None |
| Report success, moves, return, convergence, and training curves | `results/csv/`, `results/plots/`, final report | Raw CSV reconciliation and figure audit | Pass | None |
| Separate training and evaluation | Evaluation creates a fresh environment and uses deterministic policy inference; no updates | Inspect evaluation functions and reproduced saved-model metrics | Pass | None |
| Compare with Q-Learning, SARSA, and BFS; include required figures | Original and supplemental tables/plots; BFS computes exact shortest path on fixed transition graph | Independently check fixed-board BFS distances and CSV/figure references | Pass | BFS minimizes moves; RL maximizes the configured reward |
| Demonstrate random before training and a learned policy after training | `python -m smart_dice_rl play ...` prints a random game and then the selected saved-policy game | Inspect CLI and test with included demo model | Pass | None |
| Answer assignment questions, including why RL when BFS solves the deterministic board | `docs/final_report.md`, sections 6 and 14 | Compare report question list with all 10 assignment questions and research question | Pass | None |
| Optional challenges: random die, one-time snake avoidance, dynamic board, multi-player, DQN, reward optimization | DQN and reward optimization are implemented; generalized layouts studied. Other optional challenges are outside core scope | Compare assignment optional challenge list with delivered implementations | Partial by design | Random-dice, limited special actions, and multiplayer are not implemented; dynamic layouts are studied via board-aware generalization |

The assignment requires at least three approaches and explicitly names DQN as an advanced challenge; the project exceeds the minimum with five algorithms. The assignment's random-dice scenario is an optional advanced challenge, while its core formulation explicitly makes the dice choice an agent action. No results from that optional stochastic variant are mixed into the deterministic baseline.
