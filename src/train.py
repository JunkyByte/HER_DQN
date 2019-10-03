import gym
import Custom_Env.BitSwap
from src.rl_agents.dqn import DQN
import logging
import argparse
import logger as log
import numpy as np
from src.her_module.her import HER
parser = argparse.ArgumentParser()
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser.add_argument('--env', default='BitSwap-v0', type=str, help='Environment name')
parser.add_argument('--episodes', default=1_000_000, type=int, help='number of episodes')
parser.add_argument('--gamma', default=0.9, type=float, help='Discount reward factor')
parser.add_argument('--hidd_ch', default=128, type=int, help='Number of hidden units per hidden channels')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for agent training')
parser.add_argument('--eps', default=0.2, type=float, help='Chance of taking random action')
parser.add_argument('--bs', default=32, type=int, help='Batch size')
parser.add_argument('--train_interval', default=5, type=int, help='Steps of env before training')
parser.add_argument('--train_steps', default=1, type=int, help='Steps of training')
parser.add_argument('--target_int', default=100, type=int, help='Steps of training to update target network in dueling arch.')
parser.add_argument('--max_memory', default=50_000, type=int, help='Max memory')
parser.add_argument('--use_goal_state', default=True, type=boolean_string, help='Concatenate goal to state')
parser.add_argument('--use_fixed_goal', default=True, type=boolean_string, help='Use fixed goal')
parser.add_argument('--use_her', default=True, type=boolean_string, help='Use HER')
parser.add_argument('--nbits', default=20, type=int, help='Size for BitSwap environment')
parser.add_argument('--replay_k', default=4, type=int, help='Size of replayed obs from her')
parser.add_argument('--her_str', default='future', type=str, help='Strategy for her sampling')


if __name__ == '__main__':
    args = parser.parse_args()

    # Setup env
    if args.env == 'BitSwap-v0':
        env = gym.make(args.env, n=args.nbits, explicit_goal=args.use_goal_state,
                       fixed_goal=args.use_fixed_goal)
    else:
        env = gym.make(args.env)

    obs = env.reset()
    # Setup Model
    n_actions = env.action_space.n if env.action_space.shape == () else env.action_space.shape[0]
    n_state = env.observation_space.n if env.observation_space.shape == () else env.observation_space.shape[0]
    if args.use_goal_state:
        n_state += obs['desired_goal'].shape[0]

    her = None
    if args.use_her:
        her = HER(args.her_str, obs['desired_goal'].shape[0], replay_k=args.replay_k)

    dqn = DQN(state_dim=n_state,
              action_dim=n_actions,
              gamma=args.gamma,
              hidd_ch=args.hidd_ch,
              lr=args.lr,
              eps=args.eps,
              bs=args.bs,
              target_interval=args.target_int,
              max_memory=args.max_memory,
              her=her)

    train_steps = 0
    is_terminal = False
    tot_succ = 0
    for i in range(args.episodes):
        log.TB_LOGGER.step += 1
        obs = env.reset()
        if isinstance(obs, dict):
            obs, goal, reached_goal = obs['observation'], obs['desired_goal'], obs['achieved_goal']
            if args.use_goal_state:
                obs = np.concatenate([obs, goal], axis=-1)

        while True:
            action = dqn.act(obs)
            obs_new, r, is_terminal, _ = env.step(action)
            if isinstance(obs_new, dict):
                goal = obs_new['desired_goal']
                obs_new = obs_new['observation']
                if args.use_goal_state:
                    obs_new = np.concatenate([obs_new, goal], axis=-1)

            tot_succ += 1 if r != 0 else 0
            dqn.memory.store_transition(obs, obs_new, action, r, is_terminal)

            train_steps += 1
            if train_steps % args.train_interval == 0 and train_steps > 0:
                train_steps = 0
                for i in range(args.train_steps):
                    dqn.update()

            obs = obs_new
            if is_terminal:
                break

        episodes_per_epoch = 100
        if i % episodes_per_epoch == 0 and i > 0:
            log.TB_LOGGER.log_scalar(tag='Train Success', value=tot_succ / episodes_per_epoch)

        if i % 500 == 0:
            n_eval_episodes = 200
            tot_reward = 0
            for i in range(n_eval_episodes):
                obs = env.reset()
                if isinstance(obs, dict):
                    obs, goal, reached_goal = obs['observation'], obs['desired_goal'], obs['achieved_goal']
                    if args.use_goal_state:
                        obs = np.concatenate([obs, goal], axis=-1)

                while True:
                    action = dqn.act(obs, deterministic=True)
                    obs_new, r, is_terminal, _ = env.step(action)
                    if isinstance(obs_new, dict):
                        goal = obs_new['desired_goal']
                        obs_new = obs_new['observation']
                        if args.use_goal_state:
                            obs_new = np.concatenate([obs_new, goal], axis=-1)

                    #env.render()
                    tot_reward += r
                    obs = obs_new

                    if is_terminal:
                        break

            eval_succ = tot_reward / n_eval_episodes
            logging.info('Mean Reward: %s' % (eval_succ))
            log.TB_LOGGER.log_scalar(tag='Eval Success', value=eval_succ)






