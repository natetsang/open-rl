from gym.envs.registration import register


def register_envs():
    register(
        id='gridworld-v1',
        entry_point='openrl.envs.gridworld1a:GridWorld',
        max_episode_steps=200,
    )
    register(
        id='gridworld-v2',
        entry_point='openrl.envs.gridworld2a:GridWorld',
        max_episode_steps=200,
    )
    register(
        id='obstacles-cs285-v0',
        entry_point='openrl.envs.obstacles_env:Obstacles',
        max_episode_steps=500,
    )
