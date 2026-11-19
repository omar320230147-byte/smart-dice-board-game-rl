import numpy as np
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from smart_dice_rl.boards import CONFIGS, generate_board, validate_board
from smart_dice_rl.environment import SmartDiceEnv
from smart_dice_rl.generalization import BoardCollectionEnv, encode_board_state
from smart_dice_rl.reward_experiments import RewardExperimentEnv
from smart_dice_rl.run_generalization import make_boards, TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS


def test_seeded_board_generation_is_reproducible_and_valid():
    for size, snakes, ladders, seed in CONFIGS:
        a=generate_board(size,snakes,ladders,seed); b=generate_board(size,snakes,ladders,seed)
        assert a==b
        validate_board(size,a["snakes"],a["ladders"])


def test_normalized_observation_contains_board_and_state():
    b=generate_board(30,3,3,998)
    obs=encode_board_state(1,b)
    assert obs.shape==(202,) and obs.dtype==np.float32
    assert obs[0]==0 and np.isclose(obs[1],.3)


def test_board_collection_api_seed_and_checkers():
    boards=[generate_board(*cfg) for cfg in CONFIGS]
    env=BoardCollectionEnv(boards)
    gym_check_env(env,skip_render_check=True)
    sb3_check_env(env,warn=True,skip_render_check=True)
    a,_=env.reset(seed=123); b,_=env.reset(seed=123)
    assert np.array_equal(a,b)
    env.close()


def test_reward_variants_preserve_dynamics_and_potential_formula():
    board=generate_board(*CONFIGS[0])
    base=SmartDiceEnv(board["size"],board["snakes"],board["ladders"])
    wrapped=RewardExperimentEnv(base,"potential",gamma=.9)
    obs,_=wrapped.reset(seed=5)
    before=int(obs[0]); nxt,reward,term,trunc,info=wrapped.step(0)
    after=int(nxt[0]); reference=SmartDiceEnv(board["size"],board["snakes"],board["ladders"])
    reference.reset(seed=5); base_next,original,base_term,base_trunc,_=reference.step(0)
    expected=original+.9*(0 if term else after/board["size"])-before/board["size"]
    assert np.isclose(reward,expected)
    assert after==int(base_next[0]) and term==base_term and trunc==base_trunc
    wrapped.close()


def test_generalization_groups_are_disjoint_and_cover_three_sizes():
    train=make_boards(TRAIN_SEEDS); val=make_boards(VAL_SEEDS); test=make_boards(TEST_SEEDS)
    sets=[{b["seed"] for b in group} for group in (train,val,test)]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert {b["size"] for b in train}=={30,50,100}
    assert {b["size"] for b in val}=={30,50,100}
    assert {b["size"] for b in test}=={30,50,100}
