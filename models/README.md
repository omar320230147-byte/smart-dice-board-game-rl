# Model files in the public release

The five tracked root-level policies are the minimal saved artifacts used by the README demo commands: Q-Learning, SARSA, DQN, PPO, and A2C on the 30-square board (seed 17). They match the baseline experiment configuration and require no retraining for the demonstration.

The full local experiment contains hundreds of original and supplemental policy archives. Those larger collections are deliberately excluded from this repository to keep the source release focused. Their training configurations, raw result CSVs, plots, and full integrity/reload checks remain documented in `docs/final_report.md` and `results/final_audit/`. All full archives remain untouched in the local workspace and can be regenerated with the experiment commands in the README.

To play a tracked model from the project root, use the corresponding command in the README. `smart_dice_rl.play` loads the policy file matching the selected board, algorithm, and seed from this directory.
