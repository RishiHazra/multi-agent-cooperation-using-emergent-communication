"""
test file for images
"""
from tensorboardX import SummaryWriter

import os
from tqdm import tqdm
import pickle
from collections import defaultdict
import torch
import numpy as np
import utils

from agent import Agent
from arguments import Arguments

args = Arguments()

# file paths to be loaded
model_path = 'model_restricted_comm_4agents'
traits_file = open('traits_restricted_comm_4agents.pkl', 'rb')

# Initialize the summary writer
writer = SummaryWriter()


def instantiate_agents(num_step):
    agent_traits = pickle.load(traits_file)

    agent_nws = dict([(agent, Agent('cpu', agent_traits[agent], neighbors[agent]))
                      for agent in range(args.num_agents)])

    for agent in range(args.num_agents):
        agent_nws[agent].load_state_dict(torch.load(os.path.join(model_path, str(num_step) + '_' + str(agent)),
                                                    map_location=torch.device('cpu')))

    return agent_nws, agent_traits


def save_embeddings(msg, true_label, agent, episode, policy_number):
    num_time_steps = len(msg)
    writer.add_embedding(msg,
                         metadata=[agent] * num_time_steps,
                         tag='Im_msg_' + str(agent) + '_EP_' +
                             str(episode) + '_lab_' + str(true_label) + '_' + str(policy_number))


if __name__ == '__main__':
    policy_step = 40000  # model step to be loaded

    # neighbors = {0: [4,5,6,7,8,9], 1: [4,5,6,7,8,9], 2: [4,5,6,7,8,9], 3: [4,5,6,7,8,9], 4: [0,1,2,3],
    #              5: [0,1,2,3], 6: [0,1,2,3], 7: [0,1,2,3], 8: [0,1,2,3], 9: [0,1,2,3]}
    neighbors = {0:[1,2,3], 1:[0,2], 2:[0,3], 3:[0,1]}

    agents, traits_agents = instantiate_agents(policy_step)
    # calculate total reputation by summing all the 0th elements of the reputation lists
    total_reputation = torch.tensor(0.0)
    for agent_id in range(args.num_agents):
        total_reputation += traits_agents[agent_id][2][0]

    # load test set
    test_set = utils.process_dataset(evaluate=True)
    # classes in CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    num_consensus = 0  # number of times the community reaches consensus
    running_rewards = 0.0
    avg_steps, step = 0, 0  # time needed on as average to reach consensus
    # pseudo_episodes = 0

    for episode_id in tqdm(range(args.num_test_episodes)):
        # pick up one sample from test dataset
        (img, target) = test_set[episode_id]

        # if target not in [0,1,2,3,4]:
        #     continue
        # pseudo_episodes +=1

        img = img.unsqueeze(0)
        target = torch.tensor([target])

        # initialize / re-initialize all parameters\
        traits = {}
        agent_all_log_probs, agent_log_probs = {}, {}
        agent_rewards = {}
        msgs_input, msgs_broadcast = {}, {}
        save_msgs_broadcast = {}

        for agent_id in range(args.num_agents):
            num_neighbors = len(neighbors[agent_id])
            traits[agent_id] = traits_agents[agent_id].clone()
            agent_rewards[agent_id] = []
            msgs_input[agent_id] = torch.zeros(num_neighbors, args.msg_v_dim + args.msg_k_dim,
                                               dtype=torch.float32)
            msgs_broadcast[agent_id] = torch.zeros(num_neighbors, args.msg_v_dim + args.msg_k_dim,
                                                   dtype=torch.float32)
            save_msgs_broadcast[agent_id] = []

        # reset agent hidden vectors of LSTM and traits to original values
        for agent_id in range(args.num_agents):
            agents[agent_id].reset_agent(traits[agent_id])
            agents[agent_id].eval()

        # begin episode
        for step in range(args.num_steps):
            consensus = [0] * args.num_classes
            one_con = [0] * args.num_classes
            for agent_id in range(args.num_agents):

                # agents get to see the cropped image for partial observability
                cropped_image = utils.crop_image(img, str(agent_id))
                traits[agent_id], action, _, msgs_broadcast[agent_id], _ = \
                    agents[agent_id](cropped_image,
                                     msgs_input[agent_id])

                # if episode_id == 48000:
                #     resources_file.write('agent:' + str(agent_id) +
                #                          ' resource allocation: ' + str(traits[agent_id][0]) + '\n')

                # store the value of reputation parameter in the consensus
                if step != 0:
                    consensus[action.item()] += traits[agent_id][2][0].item()
                    one_con[action.item()] += 1

            # message passing
            msg_input_dict = defaultdict(list)
            for agent_id in range(args.num_agents):
                index1 = 0
                for neighbour in neighbors[agent_id]:
                    msg_input_dict[neighbour].append(msgs_broadcast[agent_id][index1].clone())
                    index1 += 1
                save_msgs_broadcast[agent_id].append(torch.cat(list(msgs_broadcast[agent_id]), dim=0).detach().tolist())

            msgs_input = dict([(agent_id, torch.stack(values)) for agent_id, values in msg_input_dict.items()])


            # define reward function
            if max(np.array(consensus)) > args.threshold and \
                    torch.tensor(consensus).argmax().unsqueeze(0) == target:
                prize = args.prize
                num_consensus += 1
                # print(consensus, ' ', one_con)
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
            print('[{}] rewards: {} steps: {} (reached consensus: {} / 2000)'.format(episode_id,
                                                                                     running_rewards / 2000,
                                                                                     avg_steps / 2000,
                                                                                     num_consensus))
            running_rewards = 0.0
            num_consensus = 0
            avg_steps = 0

        # save embeddings for 1000 episodes
        if episode_id < 1000:
            for agent_id in agents:
                save_embeddings(save_msgs_broadcast[agent_id][-2:], target.item(), agent_id, episode_id, policy_step)
            # if pseudo_episodes % 100 == 0:
            #     print('done {}'.format(pseudo_episodes))
