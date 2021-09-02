"""
MOPO - doesn't train very well. Not sure why, I don't think there are any bugs.
I tried both obstacles-cs285-v0 and Pendulum-v0 but neither of them worked.
I tried different hyperparameters for penalty type, penalty coef, rollout batch size,
num policy updates, and offline dataset size. In the paper they have an offline dataset
of 1M and their rollout batch size is either 1M (per the ppaer) or 125K (per code config file).
"""

import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Union, Tuple, List, Type
from algorithms.mopo.models import FFModel, sac_actor_fc_continuous_network, critic_fc_network
from util.utils import ReplayBuffer
from algorithms.mopo.env_utils import termination_fn, reward_fn
from algorithms.mopo.sac import SACAgent
from algorithms.mopo.online_sac import SACAgent as OnlineSACAgent
from gym.envs.registration import register
register(
    id='obstacles-cs285-v0',
    entry_point='obstacles_env:Obstacles',
    max_episode_steps=500,
)


# Set up
GAMMA = 0.99
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 3e-4
ALPHA_LEARNING_RATE = 3e-4

LOG_STD_MIN = -20
LOG_STD_MAX = 2

NUM_EPISODES = 101


class MOPOAgent:
    def __init__(self,
                 environment,
                 policy,
                 model_class: Type[FFModel],
                 replay_buffer: ReplayBuffer,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None) -> None:

        # Env vars
        self.env = environment
        self.state_dims = model_kwargs.get('state_dims')
        self.action_dims = model_kwargs.get('action_dims')
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

        # Policy
        self.policy = policy

        # Create ensemble of models
        self.ensemble_size = model_kwargs.get('ensemble_size')
        self.dyn_models = self.initialize_ensemble(model_class, model_kwargs)

        # Replay buffers
        self.replay_buffer_env = replay_buffer

        # Training vars
        self.cur_episode = 0
        self.num_model_rollouts_per_epoch = train_kwargs.get('num_model_rollouts_per_epoch')
        self.num_policy_updates_per_epoch = train_kwargs.get('num_policy_updates_per_epoch')
        self.dynamics_train_batch_size = train_kwargs.get('dynamics_train_batch_size')
        self.num_dynamics_updates = train_kwargs.get('num_dynamics_updates')

        self.rollout_length = train_kwargs.get('rollout_length')

        # Penalty
        self.reward_penalty_coef = train_kwargs.get('reward_penalty_coef')
        self.penalty_learned_var = train_kwargs.get('penalty_learned_var')

    def initialize_ensemble(self, model_class: Type[FFModel], model_kwargs: dict) -> List[FFModel]:
        """Initialize and return an ensemble of models"""
        ensemble_size = model_kwargs.get("ensemble_size")
        dyn_models = []
        for i in range(ensemble_size):
            model = model_class(ac_dim=self.action_dims,
                                ob_dim=self.state_dims,
                                n_layers=model_kwargs.get("num_hidden_layers"),
                                hidden_size=model_kwargs.get("hidden_size"))
            dyn_models.append(model)
        return dyn_models

    def train_ensemble_on_batch(self, batch: Tuple) -> np.ndarray:
        """
        Given a batch of transitions, train each model in the ensemble. For each model,
        randomly sample a subset of transitions from the batch, calculate the loss,
        and take a gradient step.
        :param batch: Batch of transitions (s,a,r,s',d)
        :return: the mean loss across the ensemble
        """
        batch_state, batch_action, _, batch_next_state, _ = batch

        # Find the number of trajectories to train each model
        num_data = batch_state.shape[0]
        num_data_per_ens = int(num_data / self.ensemble_size)

        losses = []
        # training a MB agent refers to updating the predictive model using observed state transitions
        # NOTE: each model in the ensemble is trained on a different random subset of the batch
        for model in self.dyn_models:
            # select which data points to use for this model of the ensemble
            # you might find the num_data_per_env variable defined above useful
            idxs = np.random.choice(num_data, num_data_per_ens, replace=True)
            ens_states = batch_state[idxs]
            ens_actions = batch_action[idxs]
            ens_next_states = batch_next_state[idxs]

            # Train model - do one gradient step
            with tf.GradientTape() as tape:
                loss = model.loss(obs=ens_states,
                                  acs=ens_actions,
                                  next_obs=ens_next_states,
                                  data_statistics=self.replay_buffer_env.data_statistics)
                losses.append(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return np.mean(losses)

    def train_dynamics_model(self, batch_size: int) -> Union[float, None]:
        # Get all samples from environment buffer
        if batch_size > len(self.replay_buffer_env):
            return None

        batch = self.replay_buffer_env.sample(batch_size=batch_size)

        avg_dynamics_loss = self.train_ensemble_on_batch(batch=batch)
        return float(avg_dynamics_loss)

    def set_rollout_length(self) -> None:
        # https://github.com/Xingyu-Lin/mbpo_pytorch/blob/43c8a55fa7353c6aed97525d0ecd5cb903b55377/main_mbpo.py#L162
        pass

    def get_reward_penalty(self, obs, acs, data_statistics) -> np.ndarray:
        # Get standard deviation for each model
        ensemble_mus, ensemble_stds = [], []
        for model in self.dyn_models:
            mus, stds = model.forward_pass(obs=obs, acs=acs, data_statistics=data_statistics)
            ensemble_mus.append(mus)
            ensemble_stds.append(stds)
        ensemble_mus = np.array(ensemble_mus)
        ensemble_stds = np.array(ensemble_stds)

        if self.penalty_learned_var:
            dists = np.linalg.norm(ensemble_stds, axis=2)
        else:
            mean_ensemble_mus = np.mean(ensemble_mus, axis=0)
            diffs = ensemble_mus - mean_ensemble_mus
            dists = np.linalg.norm(diffs, axis=2)
        penalties = np.amax(dists, axis=0)
        return penalties

    def rollout_policy(self, initial_states: np.ndarray) -> List[Tuple]:
        rollout = []
        states = initial_states
        for _ in range(self.rollout_length):
            # Sample dynamics model from ensemble
            model = np.random.choice(self.dyn_models)

            # Select action using policy
            actions, _ = self.policy.actor_model(states)
            next_states = model.get_prediction(obs=states,
                                               acs=actions,
                                               data_statistics=self.replay_buffer_env.data_statistics)
            # TODO >> Other implementations have access to the termination fn for each env
            #  https://github.com/Xingyu-Lin/mbpo_pytorch/blob/43c8a55fa7353c6aed97525d0ecd5cb903b55377/predict_env.py#L10
            #  https://github.com/JannerM/mbpo/blob/ac694ff9f1ebb789cc5b3f164d9d67f93ed8f129/mbpo/static/inverted_pendulum.py#L8
            if self.env.spec.id == 'obstacles-cs285-v0':
                rewards, dones = self.env.get_reward(states, actions)
            else:
                rewards = reward_fn(env=self.env, obs=states, act=actions, next_obs=next_states)
                dones = termination_fn(env=self.env, obs=states, act=actions, next_obs=next_states)

            # Get max uncertainty in the dynamics and penalize rewards
            penalties = self.get_reward_penalty(obs=states,
                                                acs=actions,
                                                data_statistics=self.replay_buffer_env.data_statistics)
            assert penalties.shape == rewards.shape
            rewards_penalized = rewards - self.reward_penalty_coef * penalties

            rollout.extend([(states[i], actions[i], rewards_penalized[i], next_states[i], dones[i])
                            for i in range(states.shape[0])])

            masks = (1 - dones).astype(bool)

            # Check if all the transitions are done
            if masks.sum() == 0:
                break

            # Only continue with transitions that aren't "done"
            states = next_states[masks]
        return rollout

    def branched_rollout(self, batch_size: int) -> None:
        # Sample s_t uniformly from D_env
        states, _, _, _, _ = self.replay_buffer_env.sample(batch_size=batch_size)

        # Perform k-step model rollout starting from s_t using policy pi_theta; Add to D_model
        transitions = self.rollout_policy(initial_states=states)
        self.policy.replay_buffer.store_transitions_batch(transitions)

    def test_agent(self) -> float:
        total_rewards = []
        for i in range(10):
            state = tf.expand_dims(tf.convert_to_tensor(self.env.reset()), 0)
            done = False
            ep_rewards = 0
            while not done:
                state = tf.reshape(state, [1, self.state_dims])
                action, _ = self.policy.actor_model(state)
                state, reward, done, _ = self.env.step(action[0])

                ep_rewards += reward
            total_rewards.append(ep_rewards)
        return float(np.mean(total_rewards))

    def train_episode(self):
        # Train on batch data D_env an ensemble of models
        if self.cur_episode == 0:
            self.replay_buffer_env.update_data_statistics()  # We only need to do this once now
            for _ in range(self.num_dynamics_updates):
                dyn_loss = self.train_dynamics_model(batch_size=self.dynamics_train_batch_size * self.ensemble_size)
            print("Dynamics model loss", dyn_loss)

        # Set rollout length for current epoch
        self.set_rollout_length()

        # Sample batch_size states from D_env and perform h-step branched rollout
        self.branched_rollout(batch_size=self.num_model_rollouts_per_epoch)

        # Use SAC to update policy
        for _ in range(self.num_policy_updates_per_epoch):
            self.policy.train_episode()

        self.cur_episode += 1


def main() -> None:
    # Create environment
    env = gym.make(args.env)
    offline_env = gym.make(args.env)

    # Set seeds
    if args.seed:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        env.seed(args.seed)
        offline_env.seed(args.seed)

    # Create helper vars for model creation
    _state_dims = len(env.observation_space.high)
    _action_dims = env.action_space.shape[0]

    # Create Replay Buffers
    buffer_env = ReplayBuffer(state_dims=_state_dims, action_dims=_action_dims)
    buffer_model = ReplayBuffer(state_dims=_state_dims, action_dims=_action_dims)

    # Instantiate optimizers
    actor_opt = tf.keras.optimizers.Adam(learning_rate=ACTOR_LEARNING_RATE)
    critic1_opt = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)
    critic2_opt = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)
    alpha_opt = tf.keras.optimizers.Adam(learning_rate=ALPHA_LEARNING_RATE)

    # Create online agent for generating data
    online_agent = OnlineSACAgent(environment=env,
                                  actor_model_fn=sac_actor_fc_continuous_network,
                                  actor_optimizer=actor_opt,
                                  critic_model_fn=critic_fc_network,
                                  critic_optimizers=(critic1_opt, critic2_opt),
                                  alpha_optimizer=alpha_opt,
                                  replay_buffer=buffer_env,
                                  model_kwargs=dict(state_dims=_state_dims,
                                                    action_dims=_action_dims,
                                                    num_hidden_layers=2,
                                                    hidden_size=256,
                                                    log_std_min=LOG_STD_MIN,
                                                    log_std_max=LOG_STD_MAX),
                                  train_kwargs=dict(policy_update_freq=2,
                                                    train_batch_size=args.policy_train_batch_size),
                                  save_dir="")

    # Data generation
    for e in range(args.online_epochs):
        _, _ = online_agent.train_episode()

    while len(online_agent.replay_buffer) < args.replay_buffer_env_size:
        online_agent.run_agent_and_add_to_buffer()

    print("Offline dataset size: ", len(buffer_env))

    # Instantiate optimizers
    actor_opt = tf.keras.optimizers.Adam(learning_rate=ACTOR_LEARNING_RATE)
    critic1_opt = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)
    critic2_opt = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)
    alpha_opt = tf.keras.optimizers.Adam(learning_rate=ALPHA_LEARNING_RATE)

    # Create offline agent
    sac_policy = SACAgent(environment=offline_env,
                          actor_model_fn=sac_actor_fc_continuous_network,
                          actor_optimizer=actor_opt,
                          critic_model_fn=critic_fc_network,
                          critic_optimizers=(critic1_opt, critic2_opt),
                          alpha_optimizer=alpha_opt,
                          replay_buffer=buffer_model,
                          model_kwargs=dict(state_dims=_state_dims,
                                            action_dims=_action_dims,
                                            num_hidden_layers=2,
                                            hidden_size=256,
                                            log_std_min=LOG_STD_MIN,
                                            log_std_max=LOG_STD_MAX),
                          train_kwargs=dict(policy_update_freq=1,  # 2 for vanilla SAC
                                            train_batch_size=args.policy_train_batch_size),
                          save_dir="")

    # Create agent
    offline_agent = MOPOAgent(environment=offline_env,
                              policy=sac_policy,
                              model_class=FFModel,
                              replay_buffer=buffer_env,
                              model_kwargs=dict(state_dims=_state_dims,
                                                action_dims=_action_dims,
                                                num_hidden_layers=2,
                                                hidden_size=256,
                                                ensemble_size=args.ensemble_size),
                              train_kwargs=dict(num_dynamics_updates=args.num_dynamics_updates,
                                                reward_penalty_coef=args.reward_penalty_coef,
                                                penalty_learned_var=args.penalty_learned_var,
                                                rollout_length=args.rollout_length,
                                                dynamics_train_batch_size=args.dynamics_train_batch_size,
                                                num_model_rollouts_per_epoch=args.num_model_rollouts_per_epoch,
                                                num_policy_updates_per_epoch=args.num_policy_updates_per_epoch,)
                              )

    # Run training
    start = time.time()
    print(f"num_dynamics_updates: {args.num_dynamics_updates} - " 
          f"num_model_rollouts_per_epoch: {args.num_model_rollouts_per_epoch} - "
          f"num_policy_updates_per_epoch: {args.num_policy_updates_per_epoch} - "
          f"reward_penalty_coef: {args.reward_penalty_coef} - "
          f"rollout_length: {args.rollout_length}")
    for e in range(args.offline_epochs):
        # Run one episode
        dmodel_size = len(offline_agent.policy.replay_buffer)
        offline_agent.train_episode()

        eval_ep_rew = offline_agent.test_agent()

        template = "EPISODE {} | eval mean ep reward: {:.2f} | D_model size: {} | total time elapsed (sec): {:.2f}"
        print(template.format(e, eval_ep_rew, dmodel_size, time.time() - start))

        # Now that we've completed training, let's plot the results
    print(f"Training time elapsed (sec): {round(time.time() - start, 2)}")

    # Now let's evaluate the trained MOPO by running it on the simulator
    print("Starting evaluation...")
    mopo_evaluation_rewards = []
    sac_evaluation_rewards = []
    for e in range(args.evaluation_epochs):
        mopo_reward = offline_agent.test_agent()
        sac_reward = online_agent.test_agent()

        mopo_evaluation_rewards.append(mopo_reward)
        sac_evaluation_rewards.append(sac_reward)

    print("Evaluation results: ")
    print("MOPO mean evaluation reward: ", np.mean(mopo_evaluation_rewards))
    print("Trained SAC mean evaluation reward: ", np.mean(sac_evaluation_rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env", type=str, default="obstacles-cs285-v0")

    parser.add_argument("--online_epochs", type=int, default=10)  # number of epochs to run online agent
    parser.add_argument("--offline_epochs", type=int, default=20)  # number of epochs to run MOPO agent
    parser.add_argument("--evaluation_epochs", type=int, default=5)  # number of epochs to evaluate MOPO agent

    parser.add_argument("--policy_train_batch_size", type=int, default=256)  # Batch size for SAC policy training
    parser.add_argument("--dynamics_train_batch_size", type=int, default=512)  # Batch size per dynamics model

    parser.add_argument("--ensemble_size", type=int, default=5)  # number of models in the ensemble
    parser.add_argument("--rollout_length", type=int, default=1)  # number of steps to take in branched rollouts

    parser.add_argument("--num_model_rollouts_per_epoch", type=int, default=1000)
    parser.add_argument("--num_policy_updates_per_epoch", type=int, default=100)

    parser.add_argument("--reward_penalty_coef", type=float, default=10)  # Penalty coefficient for rewards
    parser.add_argument("--penalty_learned_var", type=bool, default=True)  # Penalty type
    parser.add_argument("--num_dynamics_updates", type=int, default=200)
    parser.add_argument("--replay_buffer_env_size", type=int, default=10000)

    args = parser.parse_args()

    main()
