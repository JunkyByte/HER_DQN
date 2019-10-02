import gym
import Custom_Env.BitSwap
from src.rl_agents.dqn import DQN
import logging
import argparse
parser = argparse.ArgumentParser()
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


parser.add_argument('--env', default='BitSwap-v0', help='Environment name')
parser.add_argument('--episodes', default=1_000_000, type=int, help='number of episodes')
parser.add_argument('--gamma', default=0.9, type=float, help='Discount reward factor')
parser.add_argument('--hidd_ch', default=128, type=int, help='Number of hidden units per hidden channels')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for agent training')
parser.add_argument('--eps', default=0.2, type=float, help='Chance of taking random action')
parser.add_argument('--bs', default=32, type=int, help='Batch size')
parser.add_argument('--train_interval', default=5, type=int, help='Steps of env before training')
parser.add_argument('--target_int', default=1000, type=int, help='Steps of training to update target network in dueling arch.')
parser.add_argument('--max_memory', default=50_000, type=int, help='Max memory')


if __name__ == '__main__':
    args = parser.parse_args()

    # Setup env
    if args.env == 'BitSwap-v0':
        env = gym.make(args.env, n=10, explicit_goal=False)
    else:
        env = gym.make(args.env)

    # Setup Model
    n_actions = env.action_space.n if env.action_space.shape == () else env.action_space.shape[0]
    n_state = env.observation_space.n if env.observation_space.shape == () else env.observation_space.shape[0]
    dqn = DQN(state_dim=n_state,
              action_dim=n_actions,
              gamma=args.gamma,
              hidd_ch=args.hidd_ch,
              lr=args.lr,
              eps=args.eps,
              bs=args.bs,
              target_interval=args.target_int,
              max_memory=args.max_memory)

    train_steps = 0
    is_terminal = False
    max_steps = 20
    for i in range(args.episodes):
        obs = env.reset()
        if isinstance(obs, dict):
            obs, goal, reached_goal = obs['observation'], obs['desired_goal'], obs['achieved_goal']

        for s in range(max_steps):
            action = dqn.act(obs)
            obs_new, r, is_terminal, _ = env.step(action)
            if isinstance(obs_new, dict):
                obs_new = obs_new['observation']

            if s == max_steps - 1:
                is_terminal = True

            dqn.memory.store_transition(obs, obs_new, action, r, is_terminal)

            train_steps += 1
            if train_steps % args.train_interval == 0 and train_steps > 0:
                train_steps = 0
                dqn.update()

            obs = obs_new
            if is_terminal:
                break

        if i % 200 == 0:
            n_eval_episodes = 100
            tot_reward = 0
            for i in range(n_eval_episodes):
                obs = env.reset()
                if isinstance(obs, dict):
                    obs, goal, reached_goal = obs['observation'], obs['desired_goal'], obs['achieved_goal']

                for s in range(max_steps):
                    action = dqn.act(obs, deterministic=True)
                    obs_new, r, is_terminal, _ = env.step(action)
                    if isinstance(obs_new, dict):
                        obs_new = obs_new['observation']
                    #env.render()
                    tot_reward += r
                    obs = obs_new

                    if is_terminal:
                        break

            logging.info('Mean Reward: %s' % (tot_reward / n_eval_episodes))






