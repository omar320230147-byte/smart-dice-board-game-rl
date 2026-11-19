import pytest
from smart_dice_rl.environment import SmartDiceEnv

@pytest.fixture
def env(): return SmartDiceEnv(12,{6:3},{4:9},max_steps=2)

def test_normal(env):
    env.reset(seed=1); obs,r,t,tr,info=env.step(0)
    assert obs.tolist()==[2] and r==-1 and not t and not tr

def test_ladder(env):
    env.reset(); obs,r,*_=env.step(2)
    assert obs.tolist()==[9] and r==20

def test_snake(env):
    env.reset(); env.position=2; obs,r,*_=env.step(3)
    assert obs.tolist()==[3] and r==-20

def test_overshoot(env):
    env.reset(); env.position=11; obs,r,t,tr,info=env.step(1)
    assert obs.tolist()==[11] and r==-10 and not t and info['event']=='overshoot'

def test_goal(env):
    env.reset(); env.position=11; obs,r,t,tr,_=env.step(0)
    assert obs.tolist()==[12] and r==100 and t and not tr

def test_reset_and_limit(env):
    obs,_=env.reset(seed=123); assert obs.tolist()==[1]
    env.step(0); _,_,_,tr,_=env.step(0); assert tr

def test_bad_action(env):
    env.reset()
    with pytest.raises(ValueError): env.step(6)
