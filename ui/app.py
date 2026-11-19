from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import streamlit as st

from smart_dice_rl.environment import SmartDiceEnv
from smart_dice_rl.sb3_agents import ALGORITHMS
from smart_dice_rl.tabular import TabularAgent


ROOT = Path(__file__).resolve().parents[1]
BOARDS = {
    "board_1_30": ("The Short Climb", "30 squares · 3 snakes · 3 ladders"),
    "board_2_50": ("The Winding Trail", "50 squares · 5 snakes · 5 ladders"),
    "board_3_100": ("The Long Ascent", "100 squares · 8 snakes · 8 ladders"),
}
TABULAR = {"Q-Learning": "q_learning", "SARSA": "sarsa"}
DRL = {"DQN": "dqn", "PPO": "ppo", "A2C": "a2c"}
SEEDS = (17, 29, 41)

st.set_page_config(
    page_title="Smart Dice · Live Demo",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700;800&display=swap');
    :root { --ink:#18231f; --muted:#718078; --paper:#f4f5ef; --card:#fffefa;
      --green:#1f624a; --mint:#e2f2e9; --coral:#c85649; --gold:#d59c38; }
    .stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#1e352d; }
    [data-testid="stSidebar"] * { color:#f3f4ed !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color:#18231f !important; }
    [data-testid="stSidebar"] .stCaption { color:#b7c9bf !important; }
    .block-container { padding-top:2.1rem; padding-bottom:3rem; max-width:1320px; }
    .hero { display:flex; justify-content:space-between; align-items:flex-end; gap:1rem;
      padding:1.4rem 1.7rem; margin-bottom:1.25rem; border-radius:22px;
      color:#f7f7ef; background:linear-gradient(112deg,#1c3b30 0%,#245c46 70%,#37795d 100%); }
    .eyebrow { font:500 .72rem 'DM Mono',monospace; letter-spacing:.14em; text-transform:uppercase; color:#b6d6c4; }
    .hero h1 { margin:.35rem 0 .25rem; font-size:clamp(1.7rem,3vw,2.55rem); color:#fffefa; letter-spacing:-.045em; }
    .hero p { margin:0; color:#d6e6dc; font-size:.98rem; }
    .hero-mark { font-size:3.5rem; line-height:1; padding:.5rem .65rem; border-radius:18px; background:#ffffff15; }
    .section-kicker { color:var(--muted); font:500 .72rem 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.12em; }
    .metric-card { border:1px solid #e4e7dc; border-radius:15px; background:var(--card); padding:.85rem 1rem; min-height:94px; }
    .metric-label { color:var(--muted); font-size:.78rem; margin-bottom:.35rem; }
    .metric-value { color:var(--ink); font-size:1.35rem; font-weight:750; letter-spacing:-.03em; }
    .metric-sub { color:var(--muted); font-size:.73rem; margin-top:.12rem; }
    .board-shell { border:1px solid #e1e5d9; border-radius:19px; padding:12px; background:#eaeee5; }
    .board-grid { display:grid; gap:5px; }
    .tile { min-height:62px; position:relative; padding:7px 6px; overflow:hidden; border-radius:9px;
      background:#fffefa; border:1px solid #e2e5dc; display:flex; flex-direction:column; justify-content:space-between; }
    .tile:nth-child(4n+1) { background:#f9faf5; }
    .tile .num { color:#778279; font:500 .65rem 'DM Mono',monospace; }
    .tile .glyph { font-size:1.08rem; line-height:1; }
    .tile .tile-tag { position:absolute; right:5px; bottom:4px; color:#839087; font:400 .56rem 'DM Mono',monospace; }
    .tile.snake { background:#fff0ec; border-color:#efc7be; }
    .tile.snake .num,.tile.snake .glyph { color:var(--coral); }
    .tile.ladder { background:#e9f5e9; border-color:#bfdcc5; }
    .tile.ladder .num,.tile.ladder .glyph { color:var(--green); }
    .tile.player { outline:3px solid #e2aa42; outline-offset:-3px; background:#fff5d9; z-index:1; }
    .tile.goal { background:#dcefe5; border-color:#8eb89e; }
    .tile.start { background:#edf1eb; }
    .tile.player .glyph { color:#bd8421; }
    .legend { display:flex; gap:1rem; flex-wrap:wrap; color:#64736a; font-size:.76rem; padding:.6rem .2rem .1rem; }
    .legend span { display:flex; align-items:center; gap:.35rem; }
    .legend i { width:9px; height:9px; border-radius:50%; display:inline-block; }
    .move-card { padding:1rem 1.1rem; border-radius:15px; background:#fffefa; border:1px solid #e4e7dc; }
    .move-title { color:var(--ink); font-weight:750; }
    .move-detail { margin-top:.25rem; color:#56645b; font:400 .86rem 'DM Mono',monospace; }
    .event-snake { color:var(--coral); font-weight:700; }
    .event-ladder { color:var(--green); font-weight:700; }
    .event-win { color:#9a6b13; font-weight:800; }
    .stButton>button { border-radius:10px; font-weight:700; border:1px solid #dbe2d9; }
    .stButton>button[kind="primary"] { background:#1f624a; border-color:#1f624a; }
    .stProgress > div > div > div > div { background:#25805b; }
    .footer-note { color:#7b877d; font-size:.76rem; text-align:center; padding-top:1.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_board(board_key: str) -> dict:
    path = ROOT / "data" / "generated_boards" / f"{board_key}.json"
    board = json.loads(path.read_text(encoding="utf-8"))
    board["snakes"] = {int(k): int(v) for k, v in board["snakes"].items()}
    board["ladders"] = {int(k): int(v) for k, v in board["ladders"].items()}
    return board


@st.cache_resource
def load_policy(board_key: str, agent_name: str, seed: int):
    board = load_board(board_key)
    if agent_name in TABULAR:
        path = ROOT / "models" / f"{board_key}_{TABULAR[agent_name]}_seed{seed}.npz"
        if not path.exists():
            raise FileNotFoundError(f"No saved {agent_name} policy at {path}")
        agent = TabularAgent(board["size"], TABULAR[agent_name])
        agent.q = np.load(path)["q_table"]
        return agent
    if agent_name in DRL:
        path = ROOT / "models" / f"{board_key}_{DRL[agent_name]}_seed{seed}.zip"
        if not path.exists():
            raise FileNotFoundError(f"No saved {agent_name} policy at {path}")
        return ALGORITHMS[agent_name].load(str(path), device="cpu")
    return None


def available_agents(board_key: str) -> list[str]:
    available = ["Random"]
    for label, suffix in (*TABULAR.items(), *DRL.items()):
        extension = ".npz" if label in TABULAR else ".zip"
        if any((ROOT / "models" / f"{board_key}_{suffix}_seed{seed}{extension}").exists() for seed in SEEDS):
            available.append(label)
    return available


def available_seeds(board_key: str, agent_name: str) -> tuple[int, ...]:
    if agent_name == "Random":
        return SEEDS
    suffix = TABULAR.get(agent_name, DRL.get(agent_name))
    extension = ".npz" if agent_name in TABULAR else ".zip"
    return tuple(seed for seed in SEEDS if (ROOT / "models" / f"{board_key}_{suffix}_seed{seed}{extension}").exists())


def new_game(board_key: str, agent_name: str, seed: int) -> dict:
    board = load_board(board_key)
    env = SmartDiceEnv(board["size"], board["snakes"], board["ladders"])
    obs, _ = env.reset(seed=seed)
    policy = load_policy(board_key, agent_name, seed) if agent_name != "Random" else None
    return {
        "env": env,
        "policy": policy,
        "obs": obs,
        "moves": [],
        "total_reward": 0.0,
        "done": False,
        "won": False,
        "rng": np.random.default_rng(seed),
        "seed": seed,
    }


def take_turn(game: dict, agent_name: str) -> None:
    env: SmartDiceEnv = game["env"]
    state = int(game["obs"][0])
    if agent_name == "Random":
        action = int(game["rng"].integers(0, 6))
    elif agent_name in TABULAR:
        action = game["policy"].act(state, game["rng"], explore=False)
    else:
        action, _ = game["policy"].predict(game["obs"], deterministic=True)
        action = int(np.asarray(action).item())
    obs, reward, terminated, truncated, info = env.step(action)
    game["obs"] = obs
    game["total_reward"] += float(reward)
    game["done"] = bool(terminated or truncated)
    game["won"] = bool(terminated)
    game["moves"].append(
        {
            "move": len(game["moves"]) + 1,
            "from": info["old_position"],
            "dice": info["dice"],
            "target": info["target"],
            "to": info["position"],
            "event": info["event"],
            "goal": env.size,
            "reward": float(reward),
            "return": game["total_reward"],
        }
    )


def render_board(board: dict, position: int) -> str:
    size = board["size"]
    columns = 6 if size == 30 else 10
    rows = size // columns
    snake_starts = board["snakes"]
    ladder_starts = board["ladders"]
    cells = []
    for visual_row in range(rows - 1, -1, -1):
        low, high = visual_row * columns + 1, (visual_row + 1) * columns
        numbers = list(range(low, high + 1))
        if visual_row % 2:
            numbers.reverse()
        for square in numbers:
            classes = ["tile"]
            marks = []
            if square == 1:
                classes.append("start")
                marks.append("START")
            if square == size:
                classes.append("goal")
                marks.append("GOAL")
            if square in snake_starts:
                classes.append("snake")
                marks.append("SNAKE")
            elif square in ladder_starts:
                classes.append("ladder")
                marks.append("LADDER")
            if square == position:
                classes.append("player")
            symbol = "🎲" if square == position else ("🐍" if square in snake_starts else ("🪜" if square in ladder_starts else ""))
            destination = snake_starts.get(square, ladder_starts.get(square))
            destination_text = f"→{destination}" if destination is not None else (marks[0] if marks else "")
            cells.append(
                f'<div class="{" ".join(classes)}"><span class="num">{square}</span>'
                f'<span class="glyph">{symbol}</span><span class="tile-tag">{destination_text}</span></div>'
            )
    return (
        f'<div class="board-shell"><div class="board-grid" style="grid-template-columns:repeat({columns},minmax(0,1fr))">'
        + "".join(cells)
        + '</div><div class="legend"><span><i style="background:#e2aa42"></i> You are here</span>'
        + '<span><i style="background:#c85649"></i> Snake: slide back</span>'
        + '<span><i style="background:#25805b"></i> Ladder: climb up</span>'
        + '<span><i style="background:#93b89e"></i> Goal</span></div></div>'
    )


def event_copy(move: dict) -> tuple[str, str]:
    event = move["event"]
    if event == "snake":
        return f"🐍 Snake at {move['target']} sent you back to {move['to']}.", "event-snake"
    if event == "ladder":
        return f"🪜 Ladder at {move['target']} climbed to {move['to']}!", "event-ladder"
    if event == "overshoot":
        return f"↩️ Overshoot: {move['target']} is past the goal, so you stay at {move['from']}.", ""
    if move["to"] == move["target"] == move["from"] + move["dice"] and move["to"] == 1:
        return "Start your climb!", ""
    if move["to"] == move["goal"]:
        return "🏁 Goal reached!", "event-win"
    return f"Moved from {move['from']} to {move['to']}.", ""


def init_session(board_key: str, agent_name: str, seed: int) -> None:
    key = (board_key, agent_name, seed)
    if st.session_state.get("game_key") != key:
        try:
            st.session_state.game = new_game(board_key, agent_name, seed)
            st.session_state.game_key = key
            st.session_state.load_error = None
        except Exception as exc:  # Show missing/corrupt saved policy in the UI.
            st.session_state.game = None
            st.session_state.game_key = key
            st.session_state.load_error = str(exc)


with st.sidebar:
    st.markdown("<div class='eyebrow'>Interactive research demo</div>", unsafe_allow_html=True)
    st.markdown("## 🎲 Smart Dice")
    st.caption("A trained policy chooses each dice roll.")
    board_key = st.selectbox("Board", list(BOARDS), format_func=lambda k: f"{BOARDS[k][0]} · {k.split('_')[-1]} squares")
    agent_options = available_agents(board_key)
    agent_name = st.selectbox("Agent", agent_options, index=min(1, len(agent_options) - 1), help="Choose Random to show an untrained baseline, or load a saved trained policy.")
    seed_options = available_seeds(board_key, agent_name)
    seed = st.selectbox("Evaluation seed", seed_options, index=0)
    st.markdown("---")
    st.markdown("**What the agent learns**")
    st.caption("State · current square")
    st.caption("Action · choose a die value from 1 to 6")
    st.caption("Reward · goal +100 · ladder +20 · snake −20 · ordinary move −1 · overshoot −10")
    st.markdown("---")
    st.caption("Trained weights are loaded from the project’s saved model files. Evaluation is deterministic for trained policies; no policy updates happen in this demo.")
    if len(agent_options) == 1:
        st.caption("This release tracks demo policies for the 30-square board. Random play is available on every board; add other board model files to enable their trained policies.")

board = load_board(board_key)
init_session(board_key, agent_name, seed)
game = st.session_state.game

st.markdown(
    f"<div class='hero'><div><div class='eyebrow'>Course project · reinforcement learning</div>"
    f"<h1>Smart Dice Board Game</h1><p>{BOARDS[board_key][0]} &nbsp;·&nbsp; {BOARDS[board_key][1]}</p></div>"
    "<div class='hero-mark'>🎲</div></div>",
    unsafe_allow_html=True,
)

if st.session_state.load_error:
    st.error(f"Could not load this saved policy: {st.session_state.load_error}")
    st.stop()

position = int(game["obs"][0])
move_count = len(game["moves"])
status = "🏆 Goal reached" if game["won"] else ("⏱ Step limit reached" if game["done"] else "▶ In progress")
name_label = "Random baseline" if agent_name == "Random" else f"Trained {agent_name} · seed {seed}"

m1, m2, m3, m4 = st.columns(4)
metrics = [
    ("CURRENT SQUARE", f"{position} / {board['size']}", "Start 1 → goal"),
    ("MOVES", str(move_count), "One selected die per move"),
    ("CUMULATIVE REWARD", f"{game['total_reward']:+.0f}", "Assignment reward rules"),
    ("RUN STATUS", status, name_label),
]
for col, (label, value, sub) in zip((m1, m2, m3, m4), metrics):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{html.escape(value)}</div><div class="metric-sub">{html.escape(sub)}</div></div>',
            unsafe_allow_html=True,
        )

st.progress(min(position / board["size"], 1.0), text=f"Progress to goal · {position / board['size']:.0%}")
left, right = st.columns([1.5, 1], gap="large")
with left:
    st.markdown("<div class='section-kicker'>Live board</div>", unsafe_allow_html=True)
    st.markdown(render_board(board, position), unsafe_allow_html=True)

with right:
    st.markdown("<div class='section-kicker'>Agent controls</div>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        step_clicked = st.button("Roll next decision", type="primary", use_container_width=True, disabled=game["done"])
    with b2:
        autoplay_clicked = st.button("▶ Play to finish", use_container_width=True, disabled=game["done"])
    reset_clicked = st.button("↺ Reset episode", use_container_width=True)

    if reset_clicked:
        st.session_state.game_key = None
        st.rerun()
    if step_clicked:
        take_turn(game, agent_name)
        st.rerun()
    if autoplay_clicked:
        with st.spinner(f"{name_label} is playing…"):
            while not game["done"]:
                take_turn(game, agent_name)
        st.rerun()

    if game["moves"]:
        latest = game["moves"][-1]
        message, event_class = event_copy(latest)
        title = "Last decision"
        st.markdown(
            f'<div class="move-card"><div class="move-title">{title} · move {latest["move"]}</div>'
            f'<div class="move-detail">square {latest["from"]} &nbsp;→&nbsp; chose {latest["dice"]} '
            f'&nbsp;→&nbsp; landed on {latest["target"]} &nbsp;→&nbsp; ended on {latest["to"]}</div>'
            f'<div class="move-detail {event_class}">{html.escape(message)}</div>'
            f'<div class="move-detail">reward {latest["reward"]:+g} &nbsp;·&nbsp; return {latest["return"]:+g}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(f"Ready. {name_label} will choose the next die value from the current square.")
    if game["done"]:
        if game["won"]:
            st.success(f"🏁 {agent_name} reached square {board['size']} in {len(game['moves'])} moves.")
        else:
            st.warning(f"The step limit was reached at square {position}. Reset and try another agent or seed.")

with st.expander("Move-by-move decisions", expanded=bool(game["moves"])):
    if game["moves"]:
        rows = []
        for item in game["moves"][-12:][::-1]:
            rows.append(
                {
                    "Move": item["move"],
                    "From": item["from"],
                    "Die chosen": item["dice"],
                    "Landing square": item["target"],
                    "Final square": item["to"],
                    "Event": item["event"] or "—",
                    "Reward": item["reward"],
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.caption("Each decision will appear here as the game runs.")

with st.expander("How to explain the demo", expanded=False):
    st.markdown(
        """
        - **State:** the agent observes its current square.
        - **Action:** it selects a controllable dice value from 1 to 6.
        - **Transition:** the move is deterministic; overshoots stay in place, and a landing on a snake or ladder immediately changes the square.
        - **Reward:** reaching the goal gives +100; ladders +20; snakes −20; ordinary moves −1; overshoots −10.
        - **Evaluation:** trained policies are loaded from saved weights and run without learning updates. Random is the untrained comparison.
        """
    )

st.markdown('<div class="footer-note">Built from the project’s Gymnasium environment and saved policies · CPU inference · no hardcoded route</div>', unsafe_allow_html=True)
