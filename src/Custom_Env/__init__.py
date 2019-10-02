from gym.envs.registration import register

register(
    id='BitSwap-v0',
    entry_point='Custom_Env.BitSwap:BitSwapEnvironment',
    kwargs={'n' : 10, 'explicit_goal' : True},
)
