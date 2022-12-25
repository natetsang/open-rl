from gym.envs.registration import register


def register_envs():
    register(
        id='gridworld-v0',
        entry_point='openrl.envs.gridworld:GridWorld',
        max_episode_steps=500,
    )
    register(
        id='obstacles-cs285-v0',
        entry_point='openrl.envs.obstacles_env:Obstacles',
        max_episode_steps=500,
    )
