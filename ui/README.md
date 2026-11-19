# Live game demo

The Streamlit interface uses the project’s existing `SmartDiceEnv` and the saved Q-Learning, SARSA, DQN, PPO, and A2C policies. It can also run an untrained random baseline. Trained agents choose their actions from their saved policies; the demo does not hardcode a route or update model weights.

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run ui/app.py
```

Use the sidebar to select one of the three required board sizes, an agent, and an evaluation seed. Click **Roll next decision** to explain one action at a time, or **Play to finish** for a full run. The board highlights the player, snake starts, ladder starts, and goal. Each move shows the chosen die, landing square, snake/ladder result, and reward.
