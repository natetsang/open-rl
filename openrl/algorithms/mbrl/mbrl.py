"""
DQN that can either use the vanilla DQN network or a dueling DQN network
"""

import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Union, Tuple, List, Type
from algorithms.mbrl.models import FFModel
from algorithms.mbrl.utils import ReplayBufferWithNoise, plot_training_results
from envs import register_envs

register_envs()  # Register Obstacles env


class MBAgent:
    def __init__(self,
                 environment,
                 model_class: Type[FFModel],
                 replay_buffer: ReplayBufferWithNoise,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None) -> None:

        # Env vars
        self.env = environment
        self.state_dims = model_kwargs.get('state_dims')
        self.action_dims = model_kwargs.get('action_dims')
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

        # Create ensemble of models
        self.ensemble_size = model_kwargs.get('ensemble_size')
        self.dyn_models = self.initialize_ensemble(model_class, model_kwargs)

        # Replay buffer
        self.replay_buffer = replay_buffer

        # Training vars
        self.cur_episode = 0
        self.horizon = train_kwargs.get('horizon')
        self.num_sequences = train_kwargs.get('num_sequences')  # Number of action sequences
        self.max_ep_len = train_kwargs.get('max_ep_len')  # Max episode length

        # Batch size for data collection
        self.batch_size_initial = train_kwargs.get('batch_size_initial')  # Batch size for data collection in itr 0
        self.batch_size = train_kwargs.get('batch_size')  # Batch size of data collection in itr 1+
        self.eval_batch_size = train_kwargs.get('eval_batch_size')  # Batch size for eval

        # Model training vars
        self.num_agent_train_steps_per_iter = train_kwargs.get('num_agent_train_steps_per_iter')  # Grad updates per run
        self.train_batch_size = train_kwargs.get('train_batch_size')  # Batch size for training models

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

    def sample_action_sequences(self, num_sequences: int, horizon: int) -> np.ndarray:
        """
        Uniformly sample action sequences and return an array of dimensions
        (num_sequences, horizon, self.num_actions) in the range [self.action_low, self.action_high]

        @source https://github.com/cassidylaidlaw/cs285-homework/blob/master/hw4/cs285/policies/MPC_policy.py

        :param num_sequences: The number of sequences to sample
        :param horizon: The number of actions for each sequence
        :return: `num_sequences` random action sequences, each `horizon` long
        """
        random_action_sequences = self.action_low + np.random.random(
            (num_sequences, horizon, self.action_dims)) * (self.action_high - self.action_low)
        return random_action_sequences

    def get_action(self, state: np.ndarray, random: bool) -> np.ndarray:
        """
        Given a state, take an action. Either, you will take a random action, or you will take an action
        using model-predictive-control (MPC). If the latter, we randomly randomly shoot by sampling
        (N x horizon) actions. Then, for each model in the ensemble, we estimate the trajectories
        for each of the sequences and calculate the rewards. Then we average the rewards across the ensemble
        for each of the sequences. We select the sequence with the highest average reward, and take the first action
        in that sequence.

        @source https://github.com/cassidylaidlaw/cs285-homework/blob/master/hw4/cs285/policies/MPC_policy.py

        :param state: state from which to determine the next step
        :param random: boolean to determine whether to take a random action or to
                        select an action using MPC
        :return: action to take
        """
        # If random, take a single random step
        if random:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # Else, sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.num_sequences, horizon=self.horizon)

        # For each model in ensemble:
        predicted_sum_of_rewards_per_model = []
        for model in self.dyn_models:
            # Find the total rewards for each of the N action sequences
            sum_of_rewards = self.calculate_sum_of_rewards(state, candidate_action_sequences, model)
            predicted_sum_of_rewards_per_model.append(sum_of_rewards)

        # calculate mean_across_ensembles(predicted rewards)
        # find the mean total rewards across the ensembles for each of the N action sequences
        # i.e. the output will be the mean total rewards for each action sequence
        predicted_rewards = np.mean(predicted_sum_of_rewards_per_model, axis=0)  # [ens, N] --> N

        # pick the action sequence with the highest total rewards across all ensembles
        # and return the 1st element of that sequence
        best_action_sequence = candidate_action_sequences[predicted_rewards.argmax()]
        action_to_take = best_action_sequence[0]  # MPC - we only take the first action!
        return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self,
                                 obs: np.ndarray,
                                 candidate_action_sequences: np.ndarray,
                                 model: FFModel) -> np.ndarray:
        """
        For each candidate action sequence, predict a sequence of states for each dynamics model in your ensemble.
        Once you have a sequence of predicted states from each model in your ensemble,
        calculate the sum of rewards for each sequence using `self.env.get_reward(predicted_obs, actions)`
        You should sum across `self.horizon` time step. Remember that the model can process observations
        and actions in batch, which can be much faster than looping through each action sequence.

        @source https://github.com/cassidylaidlaw/cs285-homework/blob/master/hw4/cs285/policies/MPC_policy.py

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence. The array should have shape [N].
        """
        # initialize arrays
        pred_obs = np.zeros((self.num_sequences, self.horizon, self.state_dims))
        rewards = np.zeros((self.num_sequences, self.horizon))

        # Use obs as the initial state for all N action sequences. Thus, all sequences start with the same state
        pred_obs[:, 0] = np.tile(obs[None, :], (self.num_sequences, 1))

        # Iterate through each step in the horizon. Recall that it's more efficient if we process all
        # sequences at once, instead of having to loop through all the sequences too. That's why we're indexing
        # on [:, t], because the `:` means we get the value of all sequences at timestep t.
        for t in range(self.horizon):
            # Given the predicted observation and the random action, calculate the reward for this timestep
            rewards[:, t], _ = self.env.get_reward(observations=pred_obs[:, t],
                                                   actions=candidate_action_sequences[:, t])
            # Predict the next state given the current state and the random action
            if t < self.horizon - 1:
                pred_obs[:, t + 1] = model.get_prediction(obs=pred_obs[:, t],
                                                          acs=candidate_action_sequences[:, t],
                                                          data_statistics=self.replay_buffer.data_statistics)
        # Sum and return the rewards over the horizon for all sequences
        sum_of_rewards = rewards.sum(axis=1)
        assert sum_of_rewards.shape == (self.num_sequences,)
        return sum_of_rewards

    def sample_random_trajectory(self, max_path_length: int, random: bool) -> Tuple[List[Tuple], Union[int, float]]:
        """
        Sample 1 trajectory, either by taking random actions or using MPC.

        :param max_path_length: the maximum number of steps to take in the trajectory
        :param random: whether or not to sample actions randomly or using MPC
        :return:
        """
        state = tf.expand_dims(tf.convert_to_tensor(self.env.reset()), 0)
        num_steps = 0
        total_rewards = 0
        transitions = []  # transition tuples (s,a,r,s',d)
        while True:
            num_steps += 1
            action = self.get_action(state, random=random)
            action = action[0]
            next_state, reward, done, _ = self.env.step(action)
            next_state = tf.reshape(next_state, [1, self.state_dims])

            total_rewards += reward

            if done or num_steps > max_path_length:
                # self.replay_buffer.store_transition((state, action, reward, next_state, 1))
                transitions.append((state, action, reward, next_state, 1))
                break

            # self.replay_buffer.store_transition((state, action, reward, next_state, 0))
            transitions.append((state, action, reward, next_state, 0))
            state = next_state

        return transitions, total_rewards

    def sample_random_trajectories(self, batch_size: int, max_path_length: int, random: bool) -> Tuple[List, List, int]:
        """
        Sample `batch_size` trajectories by taking random actions. Each trajectory should be no longer than
        `max_path_length` steps/transitions. Note that transitions are different than trajectories!
         A transition is a tuple (s,a,r,s',d) and a trajectory is made up of 1 to `max_path_length` transitions.

        :param batch_size: The number of transitions to sample.
        :param max_path_length: The maximum steps/transitions per trajectory
        :param random: Boolean to indicate whether or not to sample actions randomly or via MPC
        :return:
        """
        num_steps_this_batch = 0
        trajectory_rewards = []
        transitions = []
        while num_steps_this_batch < batch_size:
            # print(f"  ...completed {num_steps_this_batch} of {batch_size} transitions")
            trns, rews = self.sample_random_trajectory(max_path_length, random=random)
            num_steps_this_batch += len(trns)
            trajectory_rewards.append(rews)
            # Note that we're extending, not appending, because we don't care about trajectories, we care about
            #   the transitions. If we appended, it would be ([[(tran 1), (tran 2)], ..., [(tran n), (tran n+1)]],
            #   where each sublist is a trajectory. But by extending, it's instead ([(tran 1), ..., (tran n)]
            transitions.extend(trns)
        return transitions, trajectory_rewards, num_steps_this_batch

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
                                  data_statistics=self.replay_buffer.data_statistics)
                losses.append(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return np.mean(losses)

    def train_episode(self) -> Tuple[List, int, List]:
        """
        Complete one training iteration. We first sample `use_batchsize` transitions using either random actions
        or MPC. Then we store these in the replay buffer. Then we take `num_agent_train_steps_per_iter` gradient
        updates.
        :return:
            rewards - a list of mean rewards for the trajectories across the ensemble models
            num_steps - the number of steps/transitions taken when sampling trajectories
            losses - a list of mean losses for each training step across the ensemble models
        """
        # Run base policy (random policy) get collect D = {(s,a,s')}
        if self.cur_episode == 0:
            use_batchsize = self.batch_size_initial
            select_random_action = True
        else:
            use_batchsize = self.batch_size
            select_random_action = False

        transitions, mean_traj_rewards, total_num_steps = self.sample_random_trajectories(
            batch_size=use_batchsize,
            max_path_length=self.max_ep_len,
            random=select_random_action
        )

        # Store the sampled transitions in the replay buffer
        self.replay_buffer.store_transitions_batch(transitions, noised=True)

        # Train ensemble of models
        losses = []
        for train_step in range(self.num_agent_train_steps_per_iter):
            # Sample a random batch of data from the replay buffer
            batch = self.replay_buffer.sample(batch_size=self.train_batch_size * self.ensemble_size)
            # Update all ensemble models on the batch
            avg_loss = self.train_ensemble_on_batch(batch=batch)
            losses.append(avg_loss)
        self.cur_episode += 1
        return mean_traj_rewards, total_num_steps, losses

    def run_agent(self, render=False) -> Tuple[float, int]:
        total_reward, total_steps = 0, 0
        state = self.env.reset()
        done = False

        while not done:
            if render:
                self.env.render()

            # Select action
            action = self.get_action(tf.expand_dims(state, axis=0), random=False)

            # Interact with environment
            state, reward, done, _ = self.env.step(action[0])

            # Bookkeeping
            total_reward += reward
            total_steps += 1
        return total_reward, total_steps


def main() -> None:
    # Create environment
    env = gym.make(args.env)

    # Set seeds
    if args.seed:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        env.seed(args.seed)

    # Create helper vars for model creation
    _state_dims = len(env.observation_space.high)
    _action_dims = env.action_space.shape[0]

    # Create Replay Buffer
    buffer = ReplayBufferWithNoise(state_dims=_state_dims, action_dims=_action_dims)

    # Create agent
    agent = MBAgent(environment=env,
                    model_class=FFModel,
                    replay_buffer=buffer,
                    model_kwargs=dict(state_dims=_state_dims,
                                      action_dims=_action_dims,
                                      num_hidden_layers=2,
                                      hidden_size=256,
                                      ensemble_size=args.ensemble_size),
                    train_kwargs=dict(horizon=args.horizon,
                                      num_sequences=args.num_sequences,
                                      max_ep_len=args.max_ep_len,
                                      batch_size_initial=args.batch_size_initial,
                                      batch_size=args.batch_size,
                                      train_batch_size=args.train_batch_size,
                                      eval_batch_size=args.eval_batch_size,
                                      num_agent_train_steps_per_iter=args.num_agent_train_steps_per_iter)
                    )

    # Run training
    ep_mean_rewards_history, ep_max_rewards_history, ep_min_rewards_history = [], [], []
    ep_mean_loss_history, ep_max_loss_history, ep_min_loss_history = [], [], []
    ep_steps_history = []
    ep_wallclock_history = []
    start = time.time()
    for e in range(args.epochs):
        # Run one episode
        ep_rew, ep_steps, ep_loss = agent.train_episode()
        agent.run_agent()

        # Prepare for logging
        mean_ep_rew, max_ep_rew, min_ep_rew, std_ep_rew = np.mean(ep_rew), np.max(ep_rew), np.min(ep_rew), np.std(ep_rew)
        mean_ep_loss, max_ep_loss, min_ep_loss = np.mean(ep_loss), np.max(ep_loss), np.min(ep_loss)
        ep_wallclock_history.append(time.time() - start)

        ep_mean_rewards_history.append(mean_ep_rew)
        ep_max_rewards_history.append(max_ep_rew)
        ep_min_rewards_history.append(min_ep_rew)

        ep_mean_loss_history.append(mean_ep_loss)
        ep_max_loss_history.append(max_ep_loss)
        ep_min_loss_history.append(min_ep_loss)

        ep_steps_history.append(ep_steps)

        template = "EPISODE {} | mean ep reward: {:.2f} - max ep reward: {:.2f}" \
                   " - min ep reward: {:.2f} - std ep reward: {:.2f} - mean ep loss {:.2f}"
        print(template.format(e, mean_ep_rew, max_ep_rew, min_ep_rew, std_ep_rew, mean_ep_loss))

        # Now that we've completed training, let's plot the results
    print(f"Training time elapsed (sec): {round(time.time() - start, 2)}")

    # Plot summary of results
    plot_training_results(mean_rewards_history=ep_mean_rewards_history,
                          max_rew_history=ep_max_rewards_history,
                          min_rew_history=ep_min_rewards_history,
                          mean_loss_history=ep_mean_loss_history,
                          max_loss_history=ep_max_loss_history,
                          min_loss_history=ep_min_loss_history,
                          steps_history=ep_steps_history,
                          wallclock_history=ep_wallclock_history,
                          save_dir="./results.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--env", type=str, default="obstacles-cs285-v0")
    parser.add_argument("--epochs", type=int, default=20)

    parser.add_argument("--ensemble_size", type=int, default=3)  # number of models in the ensemble
    parser.add_argument("--horizon", type=int, default=10)  # number of steps to take in random-shooting for MPC
    parser.add_argument("--num_sequences", type=int, default=1000)  # number of sequences to sample in MPC
    parser.add_argument('--max_ep_len', type=int, default=100)  # max trajectory length

    parser.add_argument('--batch_size_initial', type=int, default=5000)  # num steps/transitions to sample for itr 0
    parser.add_argument('--batch_size', '-b', type=int, default=1000)  # num steps/transitions to sample for itr 1+

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=20)  # number of grad updates per iter
    parser.add_argument('--train_batch_size', '-tb', type=int, default=512)  # training batch size per model
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=400)  # steps collected per eval iteration
    args = parser.parse_args()

    main()
