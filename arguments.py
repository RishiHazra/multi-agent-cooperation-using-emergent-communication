import argparse


def Arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--msg_k_dim', type=int, default=5,
                        help='key dimension')
    parser.add_argument('--msg_v_dim', type=int, default=5,
                        help='value dimension')
    parser.add_argument('--num_agents', type=int, default=4,
                        help='number of agents')
    parser.add_argument('--encoded_size', type=int, default=20,
                        help='output dimension of encoder')
    parser.add_argument('--prize', type=int, default=5,
                        help='reward on reaching consensus')
    parser.add_argument('--penalize', type=int, default=100,
                        help='penalize for not reaching consensus during the episode')
    parser.add_argument('--threshold', type=int, default=2,
                        help='minimum value to reach consensus')

    # ------------------------------------------------------------#

    parser.add_argument('--num_episodes', type=int, default=50000,
                        help='number of episodes')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='length of each episode')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='discount factor')

    return parser.parse_args()
