# Gymnasium environment research

**Access date: 2026-11-19**

## Search performed

I reviewed the Gymnasium environment registry documentation and third-party environment index, then searched for Gymnasium-compatible Snakes and Ladders implementations on GitHub. The registry documentation explains that discoverable environments are registered through `gymnasium.envs.registry`; the official third-party index lists game-oriented integrations but did not identify a Snakes and Ladders implementation. The GitHub search surfaced a repository named `rahuldhawani/snakes-and-ladders`, described as a jQuery exercise, rather than a Gymnasium RL environment. I found no candidate that could be verified as matching this assignment's rules and compatible with the required Gymnasium/SB3 workflow.

## Decision

The project retains its own compact implementation in `smart_dice_rl/environment.py`. This is necessary to preserve the assignment's controllable-dice action semantics and exact configurable board, reward, overshoot, transition, and truncation rules. Reusing a generic game implementation without matching those rules would undermine comparison validity. The additional generalization experiments wrap the same transition implementation and add an explicit board-aware observation rather than changing the original environment contract.

## Stable-Baselines3 integration

The environment exposes Gymnasium `Discrete(6)` actions and a valid observation space. DQN, PPO, and A2C receive the environment directly (wrapped by `Monitor` for training). The board-collection variant likewise satisfies the Gymnasium and SB3 environment checkers. Stable-Baselines3's custom-environment guidance recommends checking custom environments with `stable_baselines3.common.env_checker.check_env`; the project's tests now run this checker against all three original configurations and the generalized multi-board interface.

## References

1. Gymnasium, “Environment Registry,” https://gymnasium.farama.org/v0.26.3/api/registry/ (accessed 2026-11-19).
2. Gymnasium, “Third-Party Environments,” https://gymnasium.farama.org/v1.0.0/environments/third_party_environments/ (accessed 2026-11-19).
3. Gymnasium, “Create a Custom Environment,” https://gymnasium.farama.org/main/tutorials/gymnasium_basics/environment_creation/ (accessed 2026-11-19).
4. Stable-Baselines3, “Using Custom Environments,” https://stable-baselines3.readthedocs.io/en/v2.9.2/guide/custom_env.html (accessed 2026-11-19).
5. Stable-Baselines3, “Reinforcement Learning Tips and Tricks,” https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html (accessed 2026-11-19).
6. Rahul Dhawani, `snakes-and-ladders`, https://github.com/rahuldhawani/snakes-and-ladders (accessed 2026-11-19); unrelated jQuery game exercise, not suitable for this project.
