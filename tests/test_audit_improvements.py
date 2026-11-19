from smart_dice_rl.boards import CONFIGS, generate_board
from smart_dice_rl.baselines import bfs_distance
from smart_dice_rl.environment import SmartDiceEnv
from smart_dice_rl.experiments import evaluate
from smart_dice_rl.tabular import TabularAgent
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

def test_bfs_matches_fixed_board_distances():
    expected=(3,3,9)
    for cfg,distance in zip(CONFIGS,expected):
        board=generate_board(*cfg)
        assert bfs_distance(SmartDiceEnv(board['size'],board['snakes'],board['ladders']))==distance

def test_evaluation_does_not_update_q_table():
    board=generate_board(*CONFIGS[0]); agent=TabularAgent(board['size'],'q_learning')
    agent.q[1,0]=1.25; before=agent.q.copy()
    result=evaluate(agent,board,episodes=5,seed=1234,kind='tabular')
    assert agent.q.tolist()==before.tolist()
    assert result['eval_episodes']==5

def test_all_required_boards_pass_gymnasium_and_sb3_checkers():
    for cfg in CONFIGS:
        board=generate_board(*cfg)
        env=SmartDiceEnv(board['size'],board['snakes'],board['ladders'])
        gym_check_env(env,skip_render_check=True)
        sb3_check_env(env,warn=True)
        env.close()
