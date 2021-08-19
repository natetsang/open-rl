"""
In this script, we train a q-table on the MountainCar environment.
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # this will make us able to compare models

# Instantiate environment
env = gym.make("MountainCar-v0")

# Set hyperparameters
LEARNING_RATE = 0.1  # How much to change our q-table based on the most recent experience
DISCOUNT = 0.95  # Measure of how important we find future actions/reward vs. current
EPISODES = 2000  # How many "rounds" to play/train, with each "round" having multiple steps
SHOW_EVERY = 100  # Determines how many times to display progress
SAVE_TABLE_TO_FILE = False  # We could do this to keep track of our training, or if we wanted to select the best model
"""
Since we are using q-tables, we need to convert our continous observation space into
a discrete space. The more possible states, the larger our environment becomes and
the longer it will take to train. One of the main limitations of using q-tables is that
it can only handle "small" environments. When we have big environments, we can graduate to
DQN, or even better, other deep-RL algos. Another issue is that we need to explore every
single state in order to ensure we have the most optimal solution. In the real world,
we can't learn about every single state.

In MountainCar, a given state is represented by two dimensions: (location along x-axis, velocity)
For each dimension, we want to bucketize them into 20 equally-sized buckets.
"""

# Determine # of buckets for each dimension and the value range (aka window) for each dim within
# each bucket
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE


# Helper function to convert continuous states to discrete
def get_discrete_state(state):
    """Convert continuous state to discrete state.
    input: state = [x-coord, velocity]
    output: (bucketized x-coord, bucketized velo)
    """
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

# The higher the epsilon the higher the random action
"""
Exploitation vs. exploration
The RL-agent needs to explore new actions/states, but also learn from the past to make
well-informed decisions. If the agent only uses exploitation, it could be caught in a
locally optimal solution. If the agent only uses exploration, it will act randomly
all the time and never learn from its past. In RL, it's common to start with high
exploration, and as the agent learns, take more actions based on what it has learned.
Epsilon determines the rate at which its exploits vs. explores.

"""
# TODO >> Clean this up by putting into function
epsilon = 0.5
START_EPSILON_DECAYING = 1  # Start decay at this episode
END_EPSILON_DECAYING = EPISODES // 2 # end decay at this episode
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# We're starting with this because rewards are -1 unless you reach the top
# We are getting a 20x20x3 matrix
"""
We need to create our Q-table!
Our q-table has the shape [dim1_num_states, dim2_num_states, num_possible_actions]

For example:
- there are two dimensions to our state-space: x-position and velocity
- let's assume each dimension has 2 possible states/values
q_table = [
            [ [x1, v1], [x1, v2] ],
            [ [x2, v1], [x2, v2] ]
          ]

- but remember, for each state they have the ability to take N action
- let's assume there are 3 possible actions you can do per state

q_table = [
            [ [q_x1_v1_a1, q_x1_v1_a2, q_s1_v1_a3], [q_x1_v2_a1, q_x1_v2_a2, q_x1_v2_a3] ],
            [ [q_x2_v1_a1, q_x2_v1_a2, q_s2_v1_a3], [q_x2_v2_a1, q_x2_v2_a2, q_x2_v2_a3] ]
          ]

We could have flattened this to be just a single list of lists, but we'd have to
change a few things below to make that work. In particular, we'd have to map the actual 
values to the state #, which would be from 0 to N. If we wanted to access
a particular q-value for a given state/action, we'd do something like:
* q_table[state, action], or if we wanted to get all actions, then q_table[state, :].

We need to randomly initialize our q-table with q-values. In this particular OpenAI Gym
environment, you get -1 reward for each step until you reach the goal. When you
reach the goal, you get 0 reward and the episode ends. For this particular env,
we have 200 steps in an episode to reach the goal, after which the environment is reset.

We'll "smartly" initialze the q-values such that the `high` is 0, and we'll say
the `low` is -2. However we probably could change `low` to be something even lower...
I'm not sure how much it matters.
"""
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Create some variables that hold statistics on how the training is going
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

# Training loop
for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(f"Current episode: {episode}")
        render = True
    else:
        render = False

    # This is used for graphing. We'll sum the rewards for each episode
    episode_reward = 0

    # Get initial discrete state
    # env.reset returns the initial state to us >> env.reset() returns np.array([-1.2, -0.07])
    discrete_state = get_discrete_state(env.reset())
    done = False  # We didn't intialize at the goal, so we aren't done yet

    # Until we've reached the goal, let's take steps
    while not done:

        # Exploitation vs. exploration
        if np.random.random() > epsilon:
            # Given the current state, let's select the action that
            # has the highest q-value
            action = np.argmax(q_table[discrete_state])
        else:
            # Take a random action
            action = env.action_space.sample()
            # action = np.random.randint(0, env.action_space.n)  # Could also do this!

        # Now that we've selected the action, let's actually take it
        # Not sure why, but this function returns a tuple of 4, where the last
        # value is an empty dict (i.e. {}). So we use `_` to unpack it.
        new_state, reward, done, _ = env.step(action)

        # Let's add reward from step to the total reward for this episode
        episode_reward += reward

        # We're now in a new state, so let's discretize it
        new_discrete_state = get_discrete_state(new_state)

        # Determine whether or not we should render this step
        if render:
            env.render()

        # If we reached the goal, we can end the episode, but if not...
        if not done:
            # Let's get the q-value for the state-action pair that we just
            # previously took.
            current_q = q_table[discrete_state + (action,)]

            # Let's get the max q-value of being in the new state
            # We need to use the bellman equation to learn from the step we just took
            max_future_q = np.max(q_table[new_discrete_state])
            """
            Bellman's equation
            Update the q-value for the state-action pair we were just in
            based on transition we just made!

            Ultimately this equation is saying, let's slightly adjust our current
            q-value for the (s,a) based on what we just learned. If LEARNING_RATE = 0,
            then new_q = current_q. If LEARNING_RATE = 1, then new_q equals
            Bellman's equation.

            If DISCOUNT = 1, then we value the future q-value just as much as we
            value the reward we just received. If DISCOUNT = 0, then we don't care
            about the future q-value at all, and all we care about is the reward we
            just got. It's greedy.

            We want to balance the two. We don't want to be short sighted and only think
            about the move we just did, because that might land us in an extremely suboptimal
            spot in the future. We don't want to be far-sighted because we'll be acting
            suboptimally in the short term.
            """
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update q-value of current state/action pair that we just did
            # Note that we update the value of this state after we took an action at that state
            q_table[discrete_state + (action, )] = new_q

            # Increment our step
            discrete_state = new_discrete_state

        # I'm not really sure if we need this. In fact I don't think we need it.
        # This is saying that if we achieved our goal, we hardcode the reward
        # for being in the state and taking the action we did to 0.
        # Recall above that OpenAI defined the rewards to be -1 for each step until
        # you reach the end goal. When testing, removing this actually resulted in a lower
        # score and was more unstable!
        # I think that it will just need to train longer to achieve a better and more stable score.
        # In some envs, there won't be a known terminal state. For example, Pac-Man or Cart-Pole
        # In these envs, you can't do this special case
        # I did see a tutorial by Thomas Simonini on youtube that did not include this.
        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action, )] = 0

    # Decay epsilon since we're done with the episode
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Log details of runs so we can see how it's going and so we can plot later
    # We can print the episode stats to the CLI
    # The weird syntax is just because it means we are taking the last
    # SHOW_EVERY episodes. So if we are at episode 1500 and SHOW_EVERY = 500,
    # then we want to get data/stats from episdoes 1000-1500
    ep_rewards.append(episode_reward)
    if episode % SHOW_EVERY == 0:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f"Episode: {episode}, avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])}, max:{max(ep_rewards[-SHOW_EVERY:])}")

        # Save q_table
        if SAVE_TABLE_TO_FILE:
            np.save(f"qtables/{episode}-qtable.npy", q_table)

# Done with training!
print("Closing environment")
env.close()

# Plot training results
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.title("Q-table learning progression")
plt.legend(loc=4)
plt.show()
