"""Alternative reward wrappers; the assignment reward remains the baseline."""
from __future__ import annotations

import gymnasium as gym


class RewardExperimentEnv(gym.Wrapper):
    """Apply a named reward variant while preserving the original dynamics."""
    def __init__(self, env, mode: str, gamma: float = .9):
        super().__init__(env)
        if mode not in {"original", "reduced_ladder", "goal_focused", "potential"}:
            raise ValueError(mode)
        self.mode, self.gamma = mode, float(gamma)
        self.position = 1
        self.size = env.size

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.position = int(obs[0])
        return obs, info

    def step(self, action):
        old = self.position
        obs, reward, terminated, truncated, info = self.env.step(action)
        new = int(obs[0]); event = info.get("event")
        if self.mode == "reduced_ladder" and event == "ladder":
            reward = 5.0
        elif self.mode == "goal_focused":
            reward = 100.0 if terminated else -2.0
        elif self.mode == "potential":
            # Phi(s) = s/N, with absorbing terminal potential Phi(goal)=0.
            # PBRS: F = gamma*Phi(s') - Phi(s); see report caveat for time limits.
            phi_old = old / self.size
            phi_new = 0.0 if terminated else new / self.size
            reward = float(reward) + self.gamma * phi_new - phi_old
        self.position = new
        info["reward_mode"] = self.mode
        return obs, float(reward), terminated, truncated, info


def run_reward_experiments():
    import json, time
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from .environment import SmartDiceEnv
    from .tabular import TabularAgent
    from .experiments import evaluate
    from .improve_100 import ROOT, load_board

    out = ROOT / "results" / "improved" / "reward_experiments"
    model_dir = ROOT / "models" / "improved" / "reward_experiments"
    out.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    board = load_board(); seeds = (211, 223, 227)
    modes = ("original", "reduced_ladder", "goal_focused", "potential")
    episodes = 5000; gamma = .9; rows=[]; curves=[]
    config = {"board": "board_3_100", "modes": modes, "algorithms": ["q_learning", "sarsa"],
              "seeds": seeds, "training_episodes": episodes, "alpha": .1, "gamma": gamma,
              "epsilon_start": 1.0, "epsilon_min": .02, "epsilon_decay": .9998,
              "evaluation_episodes": 100, "evaluation_separate": True}
    (out/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for mode in modes:
        for algo in ("q_learning", "sarsa"):
            for seed in seeds:
                base = SmartDiceEnv(board['size'], board['snakes'], board['ladders'])
                env = RewardExperimentEnv(base, mode, gamma)
                agent = TabularAgent(board['size'], algo, alpha=.1, gamma=gamma,
                                     epsilon=1., epsilon_min=.02, epsilon_decay=.9998)
                started=time.perf_counter()
                checkpoints = agent.train(env, episodes, seed, progress_every=100)
                seconds=time.perf_counter()-started; env.close()
                path=model_dir/f"{mode}_{algo}_seed{seed}.npz"
                np.savez_compressed(path, q_table=agent.q)
                eval_env=RewardExperimentEnv(SmartDiceEnv(board['size'],board['snakes'],board['ladders']),mode,gamma)
                # Use an independent policy evaluator; reward wrapper changes only rewards.
                rng=np.random.default_rng(500000+seed); vals=[]
                for ep in range(100):
                    obs,_=eval_env.reset(seed=500000+seed+ep); state=int(obs[0]); total=0.; moves=0; done=False; won=False
                    while not done:
                        action=agent.act(state,rng,False)
                        obs,reward,term,trunc,info=eval_env.step(action); total+=reward; moves+=1
                        state=int(obs[0]); done=term or trunc; won=bool(info['success'])
                    vals.append((won,moves,total))
                eval_env.close(); arr=np.asarray(vals,dtype=float)
                success=float(arr[:,0].mean())
                rows.append({"board":"board_3_100","reward_mode":mode,"algorithm":algo,"seed":seed,
                             "training_episodes":episodes,"training_time_seconds":seconds,
                             "success_rate":success,"failure_rate":1-success,"avg_moves":float(arr[:,1].mean()),
                             "avg_reward":float(arr[:,2].mean()),"model_path":str(path.relative_to(ROOT))})
                curves.extend({"reward_mode":mode,"algorithm":algo,"seed":seed,**cp} for cp in checkpoints)
                print(f"reward={mode} {algo} seed={seed}: success={success:.2f}, moves={arr[:,1].mean():.1f}",flush=True)
    pd.DataFrame(rows).to_csv(out/"reward_summary.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"reward_progress.csv",index=False)
    traces=[]
    for row in rows:
        q=np.load(ROOT/row["model_path"])["q_table"]
        env=RewardExperimentEnv(SmartDiceEnv(board['size'],board['snakes'],board['ladders']),row["reward_mode"],gamma)
        obs,_=env.reset(seed=770000+int(row["seed"])); state=int(obs[0]); states=[state]; actions=[]; rewards=[]
        first_seen={state:0}; cycle_start=None; won=False
        for step in range(board['size']*4):
            action=int(np.argmax(q[state])); actions.append(action+1)
            obs,r,term,trunc,info=env.step(action); rewards.append(float(r)); state=int(obs[0]); states.append(state)
            if term: won=True; break
            if state in first_seen:
                cycle_start=first_seen[state]; break
            first_seen[state]=len(actions)
            if trunc: break
        cycle_reward=float(sum(rewards[cycle_start:])) if cycle_start is not None else np.nan
        traces.append({"reward_mode":row["reward_mode"],"algorithm":row["algorithm"],"seed":row["seed"],
                       "trace_states":" -> ".join(map(str,states)),"dice_actions":" -> ".join(map(str,actions)),
                       "transition_rewards":" -> ".join(f"{x:g}" for x in rewards),"repeated_state":states[-1] if cycle_start is not None else "",
                       "cycle_start_step":cycle_start+1 if cycle_start is not None else "",
                       "cycle_length":len(actions)-cycle_start if cycle_start is not None else 0,
                       "cycle_undiscounted_reward":cycle_reward,"reached_goal":won,"trace_steps":len(actions)})
        env.close()
    pd.DataFrame(traces).to_csv(out/"reward_cycle_diagnostics.csv",index=False)
    return rows


if __name__ == "__main__":
    run_reward_experiments()
