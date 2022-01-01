"""
This is how CS295 does actor-critics in HW3.
The main difference is that they do multiple critic-updates before
calculating the advantages and doing the actor update. In my other implementation,
I'm calculating the advantage first, and I'm only doing a single critic update
per actor update.

CS295 HW3 algo (this implementation):
- They use separate models for the actor and critic
    and I don’t think you can train a single model this way…

Train epoch
- Update critic subroutine
--- For i in num_target_updates
------  Calculate targets with critic
------  For i in num_gradient_steps:
---------- Calculate V(s) with critic
---------- Fit critic to (targets - V(s))^2
---------- Gradient step on critic

- Calculate advantages based on updated critic
- Update policy subroutine
--- Calculate PG
--- Take gradient step

My Algo:
Train epoch
- Calculate advantage
- Update critic subroutine
--- Calculate loss (Advantage^2)
--- Take gradient step on critic

- Update policy subroutine
--- Calculate PG
--- Take gradient step
"""
import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Callable, Tuple

from agents.base_agent import BaseAgent
from models.models import actor_fc_discrete_network, critic_fc_network
from util.plotting import plot_training_results


# Set up constants
GAMMA = 0.99
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.001


class VPGAgent(BaseAgent):
    def __init__(self,
                 environment: gym.Env,
                 actor_model_fn: Callable[..., tf.keras.Model],
                 actor_optimizer: tf.keras.optimizers,
                 critic_model_fn: Callable[..., tf.keras.Model],
                 critic_optimizer: tf.keras.optimizers,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None,
                 save_dir: str = None) -> None:
        # Env vars
        self.env = environment
        self.state_dims = model_kwargs.get('state_dims')
        self.num_actions = model_kwargs.get('num_actions')

        # Model vars
        self.actor_model = actor_model_fn(state_dims=self.state_dims,
                                          num_actions=self.num_actions,
                                          num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                                          hidden_size=model_kwargs.get("hidden_size"))
        self.critic_model = critic_model_fn(state_dims=self.state_dims,
                                            num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                                            hidden_size=model_kwargs.get("hidden_size"))

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # Critic training subroutine hyperparams
        self.num_target_updates = 5
        self.num_grad_steps_per_target_update = 5

        # Save directories
        self.save_dir_actor = save_dir + "_actor"
        self.save_dir_critic = save_dir + "_critic"

    def save_models(self) -> None:
        self.actor_model.save(self.save_dir_actor)
        self.critic_model.save(self.save_dir_critic)

    def load_models(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        self.actor_model = tf.keras.models.load_model(self.save_dir_actor)
        self.critic_model = tf.keras.models.load_model(self.save_dir_critic)
        return self.actor_model, self.critic_model

    def run_trajectory(self):
        ep_rewards = 0
        state = self.env.reset()
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)

        done = False
        cur_step = 0

        states, actions, next_states, rewards, masks, action_probs = [], [], [], [], [], []

        while not done:
            cur_step += 1

            # Predict action prob and take action
            action_prob = self.actor_model(state)
            action = np.random.choice(self.num_actions, p=np.squeeze(action_prob))

            next_state, reward, done, _ = self.env.step(action)
            next_state = tf.expand_dims(tf.convert_to_tensor(next_state), 0)

            # Some bookkeeping
            states.append(state)
            actions.append(tf.reshape(action, (1, 1)))
            next_states.append(next_state)
            ep_rewards += reward
            rewards.append(tf.cast(tf.reshape(reward, (1, 1)), tf.float32))
            action_probs.append(tf.convert_to_tensor([tf.expand_dims(action_prob[0][action], 0)]))
            masks.append(tf.reshape(float(1 - done), (1, 1)))

            state = next_state

        # Concat
        states = tf.concat(states, axis=0)
        actions = tf.concat(actions, axis=0)
        next_states = tf.concat(next_states, axis=0)
        rewards = tf.concat(rewards, axis=0)
        masks = tf.concat(masks, axis=0)
        action_probs = tf.concat(action_probs, axis=0)
        return ep_rewards, cur_step, states, actions, next_states, rewards, masks, action_probs

    def critic_update(self, states, next_states, rewards, masks):
        for _ in range(self.num_target_updates):
            v_next_states = self.critic_model(next_states)
            targets = rewards + GAMMA * v_next_states * masks

            for _ in range(self.num_grad_steps_per_target_update):
                with tf.GradientTape() as tape:
                    v_states = self.critic_model(states)
                    loss = tf.reduce_mean(tf.square(targets - v_states))
                grads = tape.gradient(loss, self.critic_model.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))

    def estimate_advantage(self, states, next_states, rewards, masks):
        v_states = self.critic_model(states)
        v_next_states = self.critic_model(next_states)
        q_states = rewards + GAMMA * v_next_states * masks
        advantages = q_states - v_states
        return advantages

    def train_episode(self) -> dict:
        with tf.GradientTape() as tape:
            ep_rew, cur_step, states, actions, next_states, rewards, masks, action_probs = self.run_trajectory()
            self.critic_update(states, next_states, rewards, masks)
            advantages = self.estimate_advantage(states, next_states, rewards, masks)
            actor_loss = tf.reduce_mean(-tf.math.log(action_probs) * tf.stop_gradient(advantages))
        grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))

        logs = dict()
        logs['ep_rewards'] = ep_rew
        logs['ep_steps'] = cur_step
        logs['ep_total_loss'] = actor_loss
        return logs

    def run_agent(self, render=False) -> Tuple[float, int]:
        total_reward, total_steps = 0, 0
        state = self.env.reset()
        done = False

        while not done:
            if render:
                self.env.render()

            # Select action
            action_prob = self.actor_model(tf.expand_dims(state, axis=0))
            action = np.argmax(np.squeeze(action_prob))

            # Interact with environment
            state, reward, done, _ = self.env.step(action)

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
    _num_actions = env.action_space.n

    # Create agent
    actor_opt = tf.keras.optimizers.Adam(learning_rate=ACTOR_LEARNING_RATE)
    critic_opt = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)

    agent = VPGAgent(environment=env,
                     actor_model_fn=actor_fc_discrete_network,
                     actor_optimizer=actor_opt,
                     critic_model_fn=critic_fc_network,
                     critic_optimizer=critic_opt,
                     model_kwargs=dict(state_dims=_state_dims,
                                       num_hidden_layers=2,
                                       hidden_size=128,
                                       num_actions=_num_actions),
                     save_dir=args.model_checkpoint_dir)

    # Run training
    best_mean_rewards = -float('inf')
    running_reward = 0
    ep_rewards_history = []
    ep_running_rewards_history = []
    ep_steps_history = []
    ep_loss_history = []
    ep_wallclock_history = []
    start = time.time()
    for e in range(args.epochs):
        # Train one episode
        train_logs = agent.train_episode()

        # Track progress
        ep_rew = train_logs['ep_rewards']
        ep_steps = train_logs['ep_steps']
        ep_losses = train_logs['ep_total_loss']

        running_reward = ep_rew if e == 0 else 0.05 * ep_rew + (1 - 0.05) * running_reward

        ep_rewards_history.append(ep_rew)
        ep_running_rewards_history.append(running_reward)
        ep_steps_history.append(ep_steps)
        ep_loss_history.append(ep_losses)
        ep_wallclock_history.append(time.time() - start)

        if e % 10 == 0:
            print(f"EPISODE {e} | running reward: {running_reward:.2f} - episode reward: {ep_rew:.2f}")

        latest_mean_rewards = np.mean(ep_rewards_history[-10:])
        if latest_mean_rewards > best_mean_rewards:
            best_mean_rewards = latest_mean_rewards
            agent.save_models()

        if running_reward > 195:
            print("Solved at episode {}!".format(e))
            break

    # Now that we've completed training, let's plot the results
    print(f"Training time elapsed (sec): {round(time.time() - start, 2)}")

    # Plot summary of results
    plot_training_results(rewards_history=ep_rewards_history,
                          running_rewards_history=ep_running_rewards_history,
                          steps_history=ep_steps_history,
                          loss_history=ep_loss_history,
                          wallclock_history=ep_wallclock_history,
                          save_dir="../vpg/results.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    main()
