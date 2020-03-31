"""
test file for text
"""
from tensorboardX import SummaryWriter

import os
from tqdm import tqdm
import pickle
import numpy as np
from operator import itemgetter
import torch
import torchvision

from agent import Agent
from arguments import Arguments

args = Arguments()

# file paths to be loaded
model_path = 'text_model'
traits_file = open('text_traits.pkl', 'rb')
char_map_file = open('character_mapping.pkl', 'rb')

# Initialize the summary writer
writer = SummaryWriter()


def instantiate_agents(num_step):
    agent_traits = pickle.load(traits_file)

    agent_nws = dict([(agent, Agent(agent_traits[agent]))
                      for agent in range(args.num_agents)])

    for agent in range(args.num_agents):
        agent_nws[agent].load_state_dict(torch.load(os.path.join(model_path, str(num_step) + '_' + str(agent))))

    return agent_nws, agent_traits


def save_embeddings(msg, true_label, agent, episode, policy_number):
    num_time_steps = len(msg)
    writer.add_embedding(msg,
                         metadata=[agent] * num_time_steps,
                         tag='Tx_msg_' + str(agent) + '_EP_' +
                             str(episode) + '_lab_' + str(true_label) + '_' + str(policy_number))


if __name__ == '__main__':
    policy_step = 16000  # model step to be loaded
    agents, traits_agents = instantiate_agents(policy_step)
    # calculate total reputation by summing all the 0th elements of the reputation lists
    total_reputation = torch.tensor(0.0)
    for agent_id in range(args.num_agents):
        total_reputation += traits_agents[agent_id][2][0]

    # load test set
    test_set = torchvision.datasets.CIFAR10(root='./cifar-10', train=False, download=False)
    # classes in CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # load character mapping
    character_map = pickle.load(char_map_file)

    num_consensus = 0  # number of times the community reaches consensus
    running_rewards = 0.0

    for episode_id in tqdm(range(args.num_test_episodes)):
        # pick up one sample from test dataset
        (_, target) = test_set[episode_id]
        label_class = classes[target]

        # index the target characters
        indices = list(itemgetter(*list(label_class))(character_map))
        diff = args.num_agents - len(list(label_class))
        for _ in range(diff):
            indices += [character_map['$']]  # append with special characters

        # initialize / re-initialize all parameters
        target = torch.tensor([target])
        traits = [traits_agents[agent_id].clone() for agent_id in range(args.num_agents)]
        agent_rewards = dict([(agent_id, []) for agent_id in range(args.num_agents)])
        msgs_input = dict([(agent_id, torch.zeros(args.num_agents - 1, args.msg_v_dim + args.msg_k_dim))
                           for agent_id in range(args.num_agents)])
        msgs_broadcast = dict([(agent_id, torch.zeros(args.num_agents - 1, args.msg_v_dim))
                               for agent_id in range(args.num_agents)])

        save_msgs_broadcast = dict([(agent_id, []) for agent_id in range(args.num_agents)])

        # reset agent hidden vectors of LSTM and traits to original values
        for agent_id in range(args.num_agents):
            agents[agent_id].reset_agent(traits[agent_id])
            agents[agent_id].eval()

        # begin episode
        for step in range(args.num_steps):
            consensus = [0] * args.num_classes
            for agent_id in range(args.num_agents):

                # agents get to see the a single character for partial observability

                traits[agent_id], action, _, msgs_broadcast[agent_id] = \
                    agents[agent_id](torch.tensor([indices[agent_id]], dtype=torch.float32).unsqueeze(0),
                                     msgs_input[agent_id])

                # if episode_id == 48000:
                #     resources_file.write('agent:' + str(agent_id) +
                #                          ' resource allocation: ' + str(traits[agent_id][0]) + '\n')

                # store the value of reputation parameter in the consensus
                if step != 0:
                    consensus[action.item()] += traits[agent_id][2][0].item()

            # message passing
            for agent_id in range(args.num_agents):
                index1 = 0
                for neighbour in range(args.num_agents):
                    if neighbour < agent_id:
                        msgs_input[neighbour][agent_id - 1] = msgs_broadcast[agent_id][index1].clone()
                        index1 += 1
                    elif neighbour > agent_id:
                        msgs_input[neighbour][agent_id] = msgs_broadcast[agent_id][index1].clone()
                        index1 += 1

                save_msgs_broadcast[agent_id].append(torch.cat(list(msgs_broadcast[agent_id]), dim=0).detach().tolist())

            # define reward function
            if max(np.array(consensus)) > args.threshold and \
                    torch.tensor(consensus).argmax().unsqueeze(0) == target:
                prize = args.prize
                num_consensus += 1
                for agent_id in range(args.num_agents):
                    prize_share = (traits[agent_id][2][0].detach() / total_reputation) * prize
                    agent_rewards[agent_id].append(prize_share.item())
                break
            else:
                for agent_id in range(args.num_agents):
                    prize_share = (traits[agent_id][2][0].detach() / total_reputation) * (-1)
                    agent_rewards[agent_id].append(prize_share)

        print_rewards = [torch.tensor(agent_rewards[agent_id]).sum().item()
                         for agent_id in range(args.num_agents)]

        # compute mean reward of 2,000 episodes (number of test images)
        running_rewards += np.array(print_rewards).sum()
        if episode_id % 1999 == 0 and episode_id != 0:
            print('[{}] rewards: {} (reached consensus: {} / 2000)'.format(episode_id,
                                                                            running_rewards / 2000,
                                                                            num_consensus))
            running_rewards = 0.0
            num_consensus = 0

        # save embeddings for 1000 episodes
        if episode_id < 1000:
            for agent_id in agents:
                save_embeddings(save_msgs_broadcast[agent_id], target.item(), agent_id, episode_id, policy_step)
