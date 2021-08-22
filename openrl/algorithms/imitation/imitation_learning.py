import gym
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable, Union, Tuple, List
from .models import actor_fc_discrete_network, actor_critic_fc_discrete_network
from .utils import ReplayBuffer, plot_training_results
tfd = tfp.distributions


# Set up
GAMMA = 0.99
LEARNING_RATE = 0.0001


class ImitationAgent:
    def __init__(self,
                 environment: gym.Env,
                 model_fn: Callable[..., tf.keras.Model],
                 optimizer: tf.keras.optimizers,
                 run_dagger: bool,
                 expert_policy,
                 expert_data_path,
                 replay_buffer: ReplayBuffer,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None,
                 save_dir: str = None) -> None:
        # Env vars
        self.env = environment
        self.num_inputs = model_kwargs.get('num_inputs')
        self.num_actions = model_kwargs.get('num_actions')

        num_hidden_layers = model_kwargs.get("num_hidden_layers")
        hidden_size = model_kwargs.get("hidden_size")

        # Algorithm
        self.run_dagger = run_dagger

        # Expert
        self.expert_policy = expert_policy
        self.expert_data = ImitationAgent.load_expert_data(expert_data_path)

        # Actor model
        self.model = model_fn(num_inputs=self.num_inputs,
                              num_actions=self.num_actions,
                              num_hidden_layers=num_hidden_layers,
                              hidden_size=hidden_size)

        self.optimizer = optimizer
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)  # Discrete action space only

        # Replay buffer
        self.replay_buffer = replay_buffer

        # Training vars
        self.cur_episode = 0
        self.total_steps = 0
        self.max_ep_len = train_kwargs.get("max_ep_len")
        self.batch_size = train_kwargs.get("batch_size")  # Batch size of data collection from buffer
        self.train_batch_size = train_kwargs.get('train_batch_size')  # Batch size for training models
        self.eval_batch_size = train_kwargs.get('eval_batch_size')  # Batch size for eval
        self.num_agent_train_steps_per_iter = train_kwargs.get('num_agent_train_steps_per_iter')  # Grad updates per run

    @staticmethod
    def load_expert_data(path):
        with open(path, 'rb') as f:
            expert_data = pickle.load(f)
        return expert_data

    def sample_random_trajectory(self) -> Tuple[List[Tuple], Union[int, float]]:
        """
        Sample 1 trajectory.

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
            action_prob = self.model(state)
            action = np.random.choice(self.num_actions, p=np.squeeze(action_prob))
            next_state, reward, done, _ = self.env.step(action)
            next_state = tf.reshape(next_state, [1, self.num_inputs])

            total_rewards += reward

            if done or num_steps > self.max_ep_len:
                transitions.append((state, action, reward, next_state, 1))
                break

            transitions.append((state, action, reward, next_state, 0))
            state = next_state

        return transitions, total_rewards

    def sample_n_trajectories(self) -> Tuple[List, List, int]:
        """
        Sample `self.batch_size` trajectories. Each trajectory should be no longer than
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
        while num_steps_this_batch < self.batch_size:
            traj, rews = self.sample_random_trajectory()
            num_steps_this_batch += len(traj)
            trajectory_rewards.append(rews)
            # Note that we're extending, not appending, because we don't care about trajectories, we care about
            #   the transitions. If we appended, it would be ([[(tran 1), (tran 2)], ..., [(tran n), (tran n+1)]],
            #   where each sublist is a trajectory. But by extending, it's instead ([(tran 1), ..., (tran n)]
            transitions.extend(traj)
        return transitions, trajectory_rewards, num_steps_this_batch

    def relabel_actions_with_expert(self, transitions: List[Tuple]) -> List[Tuple]:
        """
        Given a batch of transition tuples, query the Expert Policy and update the action based on
        the Expert. This is the key difference between vanilla behavioral cloning and DAgger. This
        step is equivalent to asking a human expert to label our dataset with actions the correct actions.
        """
        updated_transitions = []
        for transition in transitions:
            state, action, reward, next_state, done = transition
            action_prob, _ = self.expert_policy(state)
            expert_action = np.argmax(np.squeeze(action_prob))
            updated_transitions.append((state, expert_action, reward, next_state, done))
        return updated_transitions

    def train_episode(self) -> List:
        # Step 1: Sample trajectories
        if self.cur_episode == 0:
            # Load expert_data
            transitions = self.expert_data
        else:
            # Or sample trajectories using current policy
            transitions, _, _ = self.sample_n_trajectories()

        # Step 2: For DAgger only, ask expert policy to label data with actions
        if self.run_dagger and self.cur_episode > 0:
            transitions = self.relabel_actions_with_expert(transitions)

        # Step 3: Store the sampled transitions in the replay buffer
        self.replay_buffer.store_transitions_batch(transitions)

        # Step 4: Train model!
        losses = []
        for train_step in range(self.num_agent_train_steps_per_iter):
            # Sample a random batch of data from the replay buffer
            states, actions, _, _, _ = self.replay_buffer.sample(batch_size=self.train_batch_size)

            with tf.GradientTape() as tape:
                action_prob = self.model(states)
                loss = self.loss_fn(actions, action_prob)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            losses.append(loss)
        self.cur_episode += 1
        return losses

    def test_agent(self, render=False) -> Tuple[float, int]:
        total_steps = 0
        total_reward = 0
        state = self.env.reset()
        done = False
        while not done:
            if render:
                self.env.render()
            action_prob = self.model(tf.expand_dims(tf.convert_to_tensor(state), 0))
            action = np.argmax(np.squeeze(action_prob))
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            total_steps += 1
        return total_reward, total_steps


def main() -> None:
    # Check input params
    if args.run_dagger:
        assert args.epochs > 1, "DAgger needs more than 1 iteration of training, where each iter" \
                                "we query the expert and train"
    else:
        assert args.epochs == 1, "Vanilla behavior cloning collects expert data only once and does traditional" \
                                 "supervised learning on that dataset."

    # Create environment
    env = gym.make(args.env)

    # Set seeds
    if args.seed:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        env.seed(args.seed)

    # Create helper vars for model creation
    _num_inputs = len(env.observation_space.high)
    _num_actions = env.action_space.n

    # Create Replay Buffer
    buffer = ReplayBuffer(state_dim=_num_inputs, action_dim=1)

    # Instantiate optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Instantiate expert policy from file
    # TODO >> I think it's a bit cleaner to load the entire model instead of just the weights
    #   but I'm getting a TF error that I think was fixed in a later version. I should probably
    #   try updating the version and seeing if it fixes itself.
    expert = actor_critic_fc_discrete_network(num_inputs=_num_inputs,
                                              num_actions=_num_actions,
                                              num_hidden_layers=2,
                                              hidden_size=128)
    expert.load_weights(args.expert_policy_file)

    # Create agent
    agent = ImitationAgent(environment=env,
                           model_fn=actor_fc_discrete_network,
                           optimizer=opt,
                           replay_buffer=buffer,
                           run_dagger=args.run_dagger,
                           expert_policy=expert,
                           expert_data_path=args.expert_data,
                           model_kwargs=dict(num_inputs=_num_inputs,
                                             num_actions=_num_actions,
                                             num_hidden_layers=2,
                                             hidden_size=256),
                           train_kwargs=dict(max_ep_len=args.max_ep_len,
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
        ep_loss = agent.train_episode()
        ep_rew, ep_steps = agent.test_agent()

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

    # Let's evaluate the performance of the trained agent
    print("Beginning evaluation of trained agent!")
    eval_rew = []
    for i in range(50):
        ep_rew, ep_steps = agent.test_agent()
        eval_rew.append(ep_rew)
    print(f"Evaluation rewards: mean - {np.mean(eval_rew)} | min - {np.min(eval_rew)} | max - {np.max(eval_rew)}")

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
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument('--expert_policy_file', type=str,  default='./checkpoints/expert_model_weights')
    parser.add_argument('--expert_data', type=str, default='expert_data.pkl')
    # parser.add_argument("--run_dagger", action="store_false")
    parser.add_argument("--run_dagger", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument('--max_ep_len', type=int, default=100)  # max trajectory length

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=20)  # number of grad updates per iter
    parser.add_argument('--batch_size', type=int, default=1000)  # num steps/transitions to sample for itr 1+
    parser.add_argument('--train_batch_size', type=int, default=512)  # training batch size per model
    parser.add_argument('--eval_batch_size', type=int, default=400)  # steps collected per eval iteration
    args = parser.parse_args()

    main()
