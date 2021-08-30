"""
MOReL
- This implementation works okay. It's a bit unstable in that sometimes we get
great results, and sometimes bad results for the exact same usad_threshold, but using
different seeds.
- It's very sensitive to hyperparemeters (num_epochs, num policy updates, horizon, num rollouts per policy update)
- In general, it does seem to work. The default (usad_threshold > 10) doesn't learn very well. Adding
pessimism seems to make some improvements.
"""

import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Union, Callable, Tuple, Type, List, Dict
from dqn import DQNAgent
from .models import FFModel, dqn_fc_discrete_network, actor_critic_fc_discrete_network, fc_reward_network
from utils.utils import ReplayBuffer


# Set up constants
GAMMA = 0.99
LEARNING_RATE = 0.005
ACTOR_LOSS_WEIGHT = 1.0
CRITIC_LOSS_WEIGHT = 0.1


def compute_returns(rewards: List) -> List:
    """
    Compute the rewards-to-go, which are the cumulative rewards from t=t' to T.

    :param rewards: a list of rewards where the ith entry is the reward received at timestep t=i.
    :return: the rewards-to-go, where the ith entry is the cumulative rewards from timestep t=i to t=T,
        where T is equal to len(rewards).
    """
    discounted_rewards = []
    total_ret = 0
    for r in rewards[::-1]:
        total_ret = r + GAMMA * total_ret
        discounted_rewards.insert(0, total_ret)
    return discounted_rewards


class MOReLAgent:
    def __init__(self,
                 environment: gym.Env,
                 replay_buffer: ReplayBuffer,
                 dynamics_model_class: Type[FFModel],
                 policy_model_fn: Callable[..., tf.keras.Model],
                 policy_optimizer: tf.keras.optimizers,
                 reward_model_fn: Callable[..., tf.keras.Model] = None,
                 reward_optimizer: tf.keras.optimizers = None,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None,
                 save_dir: str = None) -> None:
        # Env vars
        self.env = environment
        self.state_dims = model_kwargs.get('state_dims')
        self.action_dims = model_kwargs.get('action_dims')
        self.num_actions = model_kwargs.get('num_actions')

        # Create ensemble of dynamics models
        self.ensemble_size = model_kwargs.get('ensemble_size')
        self.dyn_models = self.initialize_ensemble(dynamics_model_class, model_kwargs)

        # Create reward model
        self.reward_model = reward_model_fn(state_dims=self.state_dims,
                                            action_dims=self.action_dims,
                                            num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                                            hidden_size=model_kwargs.get("hidden_size"))
        self.reward_optimizer = reward_optimizer

        # Policy
        self.policy = policy_model_fn(state_dims=self.state_dims,
                                      num_actions=self.num_actions,
                                      num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                                      hidden_size=model_kwargs.get("hidden_size"))
        self.policy_optimizer = policy_optimizer

        # Replay buffer with offline data
        self.replay_buffer = replay_buffer
        self.replay_buffer.update_data_statistics()

        # Training vars
        self.cur_episode = 0
        self.horizon = train_kwargs.get('horizon')

        self.num_dynamics_train_steps_per_iter = train_kwargs.get('num_dynamics_train_steps_per_iter')
        self.dynamics_train_batch_size = train_kwargs.get('dynamics_train_batch_size')

        self.num_policy_rollouts_per_update = train_kwargs.get('num_policy_rollouts_per_update', 1)
        self.num_policy_updates_per_iter = train_kwargs.get('num_policy_updates_per_iter', 1)

        # USAD vars
        self.mean_disagreement = None
        self.std_disagreement = None
        self.max_disagreement = None
        self.usad_threshold = None
        self.halt_reward = train_kwargs.get('halt_reward', 0)  # Could have also made halt_reward a function of the min reward
        self.usad_threshold_beta = train_kwargs.get('usad_threshold_beta', 1000)

    def initialize_ensemble(self, model_class: Type[FFModel], model_kwargs: dict) -> List[FFModel]:
        """Initialize and return an ensemble of models."""
        ensemble_size = model_kwargs.get('ensemble_size')
        dyn_models = []

        for _ in range(ensemble_size):
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

        # Each model in the ensemble is trained on a different random subset of the batch
        for model in self.dyn_models:
            # Select which data points to use for this model of the ensemble
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

    def train_reward_on_batch(self, batch: Tuple) -> np.ndarray:
        """
        Given a batch of transitions, train the rewards model.
        :param batch: Batch of transitions (s,a,r,s',d)
        :return: the mean loss across the ensemble
        """
        batch_state, batch_action, batch_reward, _, _ = batch

        # Train model - do one gradient step
        with tf.GradientTape() as tape:
            pred_rew = self.reward_model([batch_state, batch_action])
            loss = tf.reduce_mean(tf.square(pred_rew - batch_reward))
        grads = tape.gradient(loss, self.reward_model.trainable_variables)
        self.reward_optimizer.apply_gradients(zip(grads, self.reward_model.trainable_variables))

        return loss

    def initialize_usad(self) -> None:
        transitions = self.replay_buffer.all()
        assert transitions is not None, "There must be data in the buffer!"
        states, actions, rewards, _, _ = transitions

        disagreements = self.calculate_disagreement(states, actions)

        self.mean_disagreement = np.mean(disagreements)
        self.std_disagreement = np.std(disagreements)
        self.max_disagreement = np.max(disagreements)
        self.usad_threshold = self.mean_disagreement + self.usad_threshold_beta * self.std_disagreement

    def calculate_disagreement(self, states, actions) -> np.ndarray:
        # Get predictions for all models
        predictions = []
        for model in self.dyn_models:
            preds = model.forward_pass(states, actions, data_statistics=self.replay_buffer.data_statistics)
            predictions.append(preds)

        # Calculate max disagreements across all models for each prediction
        disagreements = np.zeros(np.array(states).shape[0])
        for i, pred1 in enumerate(predictions):
            for j, pred2 in enumerate(predictions):
                if i < j:
                    delta = np.linalg.norm(pred1 - pred2, axis=-1)
                    disagreements = np.maximum(delta, disagreements)
        return disagreements

    def sample_initial_state(self) -> List:
        # We could just sample any state in the buffer
        # batch = self.replay_buffer.sample(batch_size=1)
        # state, _, _, _, _ = batch

        # I'm being lazy here, I could have instead kept track of the starting states
        # in the buffer.
        state = [self.env.reset()]
        return state

    def rollout_policy(self, initial_state: np.ndarray, dynamics_model: FFModel):
        trajectory = dict()
        state_trajectory, action_trajectory, action_prob_trajectory, reward_trajectory = [], [], [], []
        value_trajectory = []

        state = initial_state
        for _ in range(self.horizon):
            action_prob, values = self.policy(state)
            action = np.random.choice(self.num_actions, p=np.squeeze(action_prob))

            # Predict next state and rewards
            next_state = dynamics_model.get_prediction(obs=state,
                                                       acs=tf.convert_to_tensor([[action]], dtype=tf.float32),
                                                       data_statistics=self.replay_buffer.data_statistics)
            rewards = self.reward_model([state, tf.convert_to_tensor([[action]])])

            # Some bookkeeping
            state_trajectory.append(state)
            action_trajectory.append(tf.convert_to_tensor([[action]], dtype=tf.float32))
            action_prob_trajectory.append(tf.convert_to_tensor([tf.expand_dims(action_prob[0][action], 0)]))
            reward_trajectory.append(tf.cast(tf.reshape(rewards, (1, 1)), tf.float32))
            value_trajectory.append(values)

            state = next_state

        trajectory['states'] = tf.concat(state_trajectory, axis=0)
        trajectory['actions'] = tf.concat(action_trajectory, axis=0)
        trajectory['action_probs'] = tf.concat(action_prob_trajectory, axis=0)
        trajectory['rewards'] = tf.concat(reward_trajectory, axis=0)
        trajectory['values'] = tf.concat(value_trajectory, axis=0)
        return trajectory

    def construct_pessimistic_mdp(self, trajectory: Dict) -> Tuple[Dict, int]:
        states = trajectory['states']
        actions = trajectory['actions']
        disagreements = self.calculate_disagreement(states, actions)

        violations = np.where(disagreements > self.usad_threshold)[0]
        truncated = len(violations) > 0
        last_idx = self.horizon
        if truncated:
            last_idx = violations[0] + 1
            trajectory = {key: value[:last_idx] for (key, value) in trajectory.items()}

            # Replace last reward with USAD
            trajectory['rewards'] = trajectory['rewards'][:-1]
            trajectory['rewards'] = tf.concat([trajectory['rewards'], tf.convert_to_tensor([[self.halt_reward]], dtype=tf.float32)], axis=0)

        return trajectory, last_idx

    def train_policy(self):
        # Rollout policies
        trajectories = []
        losses = []

        num_rollouts = self.num_policy_rollouts_per_update // self.ensemble_size
        with tf.GradientTape() as tape:
            # Gather trajectories by rolling out policy on dynamics model
            for k in range(num_rollouts):
                for model in self.dyn_models:
                    # Sample an initial state
                    init_state = self.sample_initial_state()

                    # Rollout policy using dynamics, reward and policy models
                    traj = self.rollout_policy(initial_state=np.array(init_state), dynamics_model=model)
                    trajectories.append(traj)

            # Now go through each trajectory and calculate the actor-critic loss
            last_idxs = []
            for trajectory in trajectories:
                # Truncate based on USAD
                trajectory, last_idx = self.construct_pessimistic_mdp(trajectory)
                last_idxs.append(last_idx)

                # Calculate rewards
                returns = compute_returns(trajectory['rewards'])
                trajectory['discounted_returns'] = tf.expand_dims(tf.concat(returns, axis=0), axis=1)

                # Calculate advantages
                trajectory['advantages'] = trajectory['discounted_returns'] - trajectory['values']
                advantages = trajectory['advantages']

                # Calculate losses
                action_prob_trajectory = trajectory['action_probs']
                actor_loss = -tf.math.log(action_prob_trajectory) * tf.stop_gradient(advantages)
                critic_loss = tf.square(advantages)
                total_loss = tf.reduce_mean(actor_loss * ACTOR_LOSS_WEIGHT +
                                            critic_loss * CRITIC_LOSS_WEIGHT)
                losses.append(total_loss)

            mean_loss = tf.reduce_mean(losses)

        # Backpropagate loss
        grads = tape.gradient(mean_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        return mean_loss, np.mean(last_idxs)

    def train_episode(self):
        # Step 1: Train ensemble of models
        dyn_losses, rew_losses = [], []
        for dynamics_train_step in range(self.num_dynamics_train_steps_per_iter):
            # Sample a random batch of data from the replay buffer
            batch = self.replay_buffer.sample(batch_size=self.dynamics_train_batch_size * self.ensemble_size)

            # Train all ensemble models on the batch
            mean_ensemble_loss = self.train_ensemble_on_batch(batch=batch)
            dyn_losses.append(mean_ensemble_loss)

            # Train reward model on batch
            rew_loss = self.train_reward_on_batch(batch=batch)
            rew_losses.append(rew_loss)

        # Step 2: Compute and initialize variables for USAD
        self.initialize_usad()

        # Step 3: Train policy
        policy_losses = []
        for policy_train_step in range(self.num_policy_updates_per_iter):
            policy_loss, last_idxs = self.train_policy()
            print(f"POLICY TRAIN STEP: {policy_train_step} -- POLICY LOSS: {policy_loss} -- LAST IDX: {last_idxs}")
            policy_losses.append(policy_loss)

        info = dict()
        info['dynamics model loss'] = np.mean(dyn_losses)
        info['rewards model loss'] = np.mean(rew_losses)
        info['policy loss'] = np.mean(policy_losses)
        self.cur_episode += 1
        return info

    def test_agent(self, render=False) -> Union[float, int]:
        total_reward = 0
        state = self.env.reset()
        done = False
        while not done:
            if render:
                self.env.render()
            action_prob, _ = self.policy(tf.expand_dims(tf.convert_to_tensor(state), 0))
            action = np.argmax(np.squeeze(action_prob))
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
        return total_reward


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
    _is_discrete_action = type(env.action_space) == gym.spaces.discrete.Discrete
    _num_actions = env.action_space.n if _is_discrete_action else env.action_space.shape[0]
    _action_dims = 1 if _is_discrete_action else env.action_space.shape[0]

    # Create Replay Buffer
    buffer = ReplayBuffer(state_dims=_state_dims, action_dims=_action_dims)

    # Select network architecture
    online_model_func = dqn_fc_discrete_network

    dynamics_model_class = FFModel
    policy_model_func = actor_critic_fc_discrete_network
    reward_model_func = fc_reward_network

    # Instantiate optimizers
    online_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    offline_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    reward_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Create online agent for generating data
    online_agent = DQNAgent(environment=env,
                            model_fn=online_model_func,
                            optimizer=online_opt,
                            replay_buffer=buffer,
                            model_kwargs=dict(state_dims=_state_dims,
                                              num_actions=_num_actions,
                                              num_hidden_layers=2,
                                              hidden_size=256),
                            train_kwargs=dict(target_update_freq=20,
                                              use_polyak=False),
                            save_dir=None)

    # Data generation
    for e in range(args.online_epochs):
        _, _ = online_agent.train_episode()

    print("Offline dataset size: ", len(buffer.size))

    print(f"Starting training (beta = {args.usad_threshold_beta})")
    # Create offline agent
    offline_agent = MOReLAgent(environment=offline_env,
                               dynamics_model_class=dynamics_model_class,
                               policy_model_fn=policy_model_func,
                               policy_optimizer=offline_opt,
                               reward_model_fn=reward_model_func,
                               reward_optimizer=reward_opt,
                               replay_buffer=buffer,  # Use buffer with prepopulated data
                               model_kwargs=dict(state_dims=_state_dims,
                                                 action_dims=_action_dims,
                                                 num_actions=_num_actions,
                                                 num_hidden_layers=2,
                                                 hidden_size=256,
                                                 ensemble_size=args.ensemble_size),
                               train_kwargs=dict(horizon=args.horizon,
                                                 dynamics_train_batch_size=args.dynamics_train_batch_size,
                                                 num_dynamics_train_steps_per_iter=args.num_dynamics_train_steps_per_iter,
                                                 num_policy_rollouts_per_update=args.num_policy_rollouts_per_update,
                                                 num_policy_updates_per_iter=args.num_policy_updates_per_iter,
                                                 usad_threshold_beta=args.usad_threshold_beta,
                                                 halt_reward=args.halt_reward),
                               save_dir=args.model_checkpoint_dir)

    # Run offline training
    running_dynamics_loss, running_rewards_loss, running_policy_loss = 0, 0, 0
    ep_dynamics_running_loss_history = []
    ep_rewards_running_loss_history = []
    ep_policy_running_loss_history = []
    start = time.time()
    for e in range(args.offline_epochs):
        log = offline_agent.train_episode()

        dynamics_model_loss = log['dynamics model loss']
        rewards_model_loss = log['rewards model loss']
        policy_loss = log['policy loss']

        # Track progress
        if e == 0:
            running_dynamics_loss = dynamics_model_loss
            running_rewards_loss = rewards_model_loss
            running_policy_loss = policy_loss
        else:
            running_dynamics_loss = 0.05 * dynamics_model_loss + (1 - 0.05) * running_dynamics_loss
            running_rewards_loss = 0.05 * rewards_model_loss + (1 - 0.05) * running_rewards_loss
            running_policy_loss = 0.05 * policy_loss + (1 - 0.05) * running_policy_loss

        # Print results
        template = "policy running loss: {:.2f} | episode loss: {:.2f} at episode {}"
        print(template.format(running_policy_loss, policy_loss, e))
        template = "dynamics running loss: {:.2f} | episode loss: {:.2f} at episode {}"
        print(template.format(running_dynamics_loss, dynamics_model_loss, e))
        template = "reward running loss: {:.2f} | episode loss: {:.2f} at episode {}"
        print(template.format(running_rewards_loss, rewards_model_loss, e))
    print(f"Training time elapsed (sec): {round(time.time() - start, 2)}")

    # Now let's evaluate the trained CQL by running it on the simulator
    print("Starting evaluation...")
    morel_evaluation_rewards = []
    dqn_evaluation_rewards = []
    for e in range(args.evaluation_epochs):
        morel_reward = offline_agent.test_agent()
        dqn_reward = online_agent.test_agent()

        morel_evaluation_rewards.append(morel_reward)
        dqn_evaluation_rewards.append(dqn_reward)

    print("Evaluation results: ")
    print(f"MOReL mean evaluation reward (beta = {args.usad_threshold_beta}) : {np.mean(morel_evaluation_rewards)}")
    print("Trained DQN mean evaluation reward: ", np.mean(dqn_evaluation_rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")

    parser.add_argument("--online_epochs", type=int, default=80)  # number of epochs to run online agent
    parser.add_argument("--offline_epochs", type=int, default=1)  # number of epochs to run MOReL agent
    parser.add_argument("--evaluation_epochs", type=int, default=100)  # number of epochs to evaluate MOReL agent

    parser.add_argument("--ensemble_size", type=int, default=4)  # number of dynamics models in the ensemble
    parser.add_argument("--horizon", type=int, default=25)  # number of steps to take model-based rollouts
    parser.add_argument("--usad_threshold_beta", type=float, default=0.8)  # Beta for calculating USAD threshold
    parser.add_argument("--halt_reward", type=float, default=-5.0)  # Beta for calculating USAD threshold

    parser.add_argument('--num_dynamics_train_steps_per_iter', type=int, default=1000)  # number of dynamics grad updates per iter
    parser.add_argument('--dynamics_train_batch_size', type=int, default=128)  # training batch size per dynamics model
    parser.add_argument('--num_policy_updates_per_iter', type=int, default=50)  # number of policy grad updates per iter
    parser.add_argument('--num_policy_rollouts_per_update', type=int, default=10)

    args = parser.parse_args()

    main()
