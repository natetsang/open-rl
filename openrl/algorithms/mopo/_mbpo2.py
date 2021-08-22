"""MBPO"""

import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Union, Tuple, List, Type
from .models import FFModel, sac_actor_fc_continuous_network, critic_fc_network
from .utils import ReplayBuffer, plot_training_results
from sac import SACAgent
from gym.envs.registration import register
register(
    id='obstacles-cs285-v0',
    entry_point='obstacles_env:Obstacles',
    max_episode_steps=500,
)

tf.keras.backend.set_floatx('float32')

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
        self.num_inputs = model_kwargs.get('num_inputs')
        self.num_actions = model_kwargs.get('num_actions')
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
        self.init_exploration_steps = train_kwargs.get('init_exploration_steps')
        self.num_env_steps_per_epoch = train_kwargs.get('num_env_steps_per_epoch')
        self.num_model_rollouts_per_env_step = train_kwargs.get('num_model_rollouts_per_env_step')
        self.num_policy_updates_per_env_step = train_kwargs.get('num_policy_updates_per_env_step')
        self.dynamics_train_batch_size = train_kwargs.get('dynamics_train_batch_size')

        # Policy vars for online rollout
        self.cur_state = None
        self.path_steps = 0

        self.rollout_length = train_kwargs.get('rollout_length')
        self.max_ep_len = train_kwargs.get('max_ep_len')  # Max episode length

    def initialize_ensemble(self, model_class: Type[FFModel], model_kwargs: dict) -> List[FFModel]:
        """Initialize and return an ensemble of models"""
        ensemble_size = model_kwargs.get("ensemble_size")
        dyn_models = []
        for i in range(ensemble_size):
            model = model_class(ac_dim=self.num_actions,
                                ob_dim=self.num_inputs,
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
        # TODO >> I should update the
        self.replay_buffer_env.update_data_statistics()

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
            rewards, dones = self.env.get_reward(states, actions)

            rollout.extend([(states[i], actions[i], rewards[i], next_states[i], dones[i])
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

    def sample_action(self) -> Tuple:
        if self.cur_state is None:
            self.cur_state = tf.expand_dims(tf.convert_to_tensor(self.env.reset()), 0)
        state = self.cur_state

        action, _ = self.policy.actor_model(state)
        next_state, reward, done, _ = self.env.step(action[0])
        next_state = tf.reshape(next_state, [1, self.num_inputs])

        if done or self.path_steps > self.max_ep_len:
            self.cur_state = None
            self.path_steps = 0
        else:
            self.cur_state = next_state
            self.path_steps += 1

        return state, action, reward, next_state, done

    def test_agent(self) -> float:
        total_rewards = []
        for i in range(10):
            state = tf.expand_dims(tf.convert_to_tensor(self.env.reset()), 0)
            done = False
            ep_rewards = 0
            while not done:
                state = tf.reshape(state, [1, self.num_inputs])
                action, _ = self.policy.actor_model(state)
                state, reward, done, _ = self.env.step(action[0])

                ep_rewards += reward
            total_rewards.append(ep_rewards)
        return float(np.mean(total_rewards))

    def exploration_before_start(self) -> None:
        init_transitions = []
        for _ in range(self.init_exploration_steps):
            state, action, reward, next_state, done = self.sample_action()
            init_transitions.append((state, action, reward, next_state, done))
        self.replay_buffer_env.store_transitions_batch(init_transitions)

    def train_episode(self):
        # Initial exploration
        if self.cur_episode == 0:
            self.exploration_before_start()

        # Train ensemble of models
        avg_dynamics_loss = self.train_dynamics_model(batch_size=self.dynamics_train_batch_size * self.ensemble_size)

        # Set rollout length for current epoch
        self.set_rollout_length()

        for _ in range(self.num_env_steps_per_epoch):
            # Take action in environment according to pi_theta; Add to D_env
            state, action, reward, next_state, done = self.sample_action()
            self.replay_buffer_env.store_transition((state, action, reward, next_state, done))

            # Sample s_t from D_env; Perform k-step rollout starting from s_t using policy; Add transitions to D_model
            self.branched_rollout(batch_size=self.num_model_rollouts_per_env_step)

            # Update policy params on model data
            for _ in range(self.num_policy_updates_per_env_step):
                self.policy.train_episode()

        self.cur_episode += 1
        return avg_dynamics_loss


def main() -> None:
    # Create environment
    env = gym.make(args.env)

    # Set seeds
    if args.seed:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        env.seed(args.seed)

    # Create helper vars for model creation
    _num_inputs = len(env.observation_space.high)
    _num_actions = env.action_space.shape[0]

    # Create Replay Buffers
    buffer_env = ReplayBuffer(state_dim=_num_inputs, action_dim=_num_actions)
    buffer_model = ReplayBuffer(state_dim=_num_inputs, action_dim=_num_actions)

    actor_opt = tf.keras.optimizers.Adam(learning_rate=ACTOR_LEARNING_RATE)
    critic1_opt = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)
    critic2_opt = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)
    alpha_opt = tf.keras.optimizers.Adam(learning_rate=ALPHA_LEARNING_RATE)

    # Create agent
    sac_policy = SACAgent(environment=env,
                          actor_model_fn=sac_actor_fc_continuous_network,
                          actor_optimizer=actor_opt,
                          critic_model_fn=critic_fc_network,
                          critic_optimizers=(critic1_opt, critic2_opt),
                          alpha_optimizer=alpha_opt,
                          replay_buffer=buffer_model,
                          model_kwargs=dict(num_inputs=_num_inputs,
                                            num_actions=_num_actions,
                                            num_hidden_layers=2,
                                            hidden_size=256,
                                            log_std_min=LOG_STD_MIN,
                                            log_std_max=LOG_STD_MAX),
                          train_kwargs=dict(policy_update_freq=1,  # 2 for vanilla SAC

                                            train_batch_size=args.policy_train_batch_size),
                          save_dir="")

    # Create agent
    agent = MOPOAgent(environment=env,
                      policy=sac_policy,
                      model_class=FFModel,
                      replay_buffer=buffer_env,
                      model_kwargs=dict(num_inputs=_num_inputs,
                                        num_actions=_num_actions,
                                        num_hidden_layers=2,
                                        hidden_size=256,
                                        ensemble_size=args.ensemble_size),
                      train_kwargs=dict(init_exploration_steps=args.init_exploration_steps,
                                        rollout_length=args.rollout_length,
                                        max_ep_len=args.max_ep_len,
                                        dynamics_train_batch_size=args.dynamics_train_batch_size,
                                        num_env_steps_per_epoch=args.num_env_steps_per_epoch,
                                        num_model_rollouts_per_env_step=args.num_model_rollouts_per_env_step,
                                        num_policy_updates_per_env_step=args.num_policy_updates_per_env_step)
                      )

    # Run training
    start = time.time()
    print(f"num_env_steps_per_epoch: {args.num_env_steps_per_epoch} - "
          f"num_model_rollouts_per_env_step: {args.num_model_rollouts_per_env_step} - "
          f"num_policy_updates_per_env_step: {args.num_policy_updates_per_env_step}")
    for e in range(args.epochs):
        # Run one episode
        dynamics_loss = agent.train_episode()

        eval_ep_rew = agent.test_agent()

        template = "EPISODE {} | dynamics loss: {:.2f} | eval mean ep reward: {:.2f} | total time elapsed (sec): {:.2f}"
        print(template.format(e, dynamics_loss, eval_ep_rew, time.time() - start))

        # Now that we've completed training, let's plot the results
    print(f"Training time elapsed (sec): {round(time.time() - start, 2)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env", type=str, default="obstacles-cs285-v0")
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--ensemble_size", type=int, default=3)  # number of models in the ensemble
    parser.add_argument("--init_exploration_steps", type=int, default=5000)  # number of steps to take before training
    parser.add_argument("--rollout_length", type=int, default=1)  # number of steps to take in branched rollouts
    parser.add_argument('--max_ep_len', type=int, default=100)  # max trajectory length

    parser.add_argument("--num_env_steps_per_epoch", type=int, default=500)  # 1000 in MBPO paper
    parser.add_argument("--num_model_rollouts_per_env_step", type=int, default=100)  # 400 in MBPO paper
    parser.add_argument("--num_policy_updates_per_env_step", type=int, default=10)  # 20 in MBPO paper

    parser.add_argument("--policy_train_batch_size", type=int, default=256)  # Batch size for SAC policy training
    parser.add_argument("--dynamics_train_batch_size", type=int, default=512)  # Batch size per dynamicsc model

    args = parser.parse_args()

    main()
