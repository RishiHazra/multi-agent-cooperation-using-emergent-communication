"""
train file for images
"""

import gc
import pickle
import os
import numpy as np
import torch
# from torch.nn.utils import clip_grad_value_
import torch.nn as nn
import torch.optim as optim

import utils
from agent import Agent
from arguments import Arguments

args = Arguments()
model_path = 'model_with_L1'
log_file = open('log_with_L1.txt', 'w')
resources_file = open('resource_allocation_with_L1.txt', 'w')
traits_file = open('traits_with_L1.pkl', 'wb')


def instantiate_agents():
    """
    :return: agents, traits {z_i, u_i, alpha_i}
             z_i : resources at disposal of agent i
             u_i : intolerance for dominance of agent i
             alpha_i : reputation of agent i
    """
    agent_traits = dict((agent_ind, list()) for agent_ind in range(args.num_agents))
    for agent_ind in range(args.num_agents):
        # resources at disposal
        agent_traits[agent_ind] = torch.softmax(
            torch.randint(1, args.num_agents, (args.num_agents - 1,), dtype=torch.float32,
                          requires_grad=False).unsqueeze(0), dim=-1)
        # intolerance for dominance (negative value)
        agent_traits[agent_ind] = torch.cat((agent_traits[agent_ind], -torch.rand(1, args.num_agents - 1,
                                                                                  dtype=torch.float32,
                                                                                  requires_grad=False)))
        # reputation
        agent_traits[agent_ind] = torch.cat((agent_traits[agent_ind],
                                             torch.randint(1, 2, (1,), dtype=torch.float32, requires_grad=False).repeat(
                                                 args.num_agents - 1).unsqueeze(0)))
        print(agent_traits[agent_ind][1])
    agent_nws = dict([(agent_ind, Agent(agent_traits[agent_ind])) for agent_ind in range(args.num_agents)])
    return agent_nws, agent_traits


def compute_loss(target_class, all_log_probs, rewards_agents, log_probs):
    """
    :param target_class: target label
    :param all_log_probs: dict of all log_probs of actions taken by all agents
    :param rewards_agents: dict of log_probs of actions taken by all agents
    :param log_probs: dict of log_probs of actions taken by all agents
    :return: classification + RL loss
    """
    classification_loss = nn.CrossEntropyLoss()
    # Compute loss for each agent
    for agent_ind in range(args.num_agents):

        # Compute cumulative rewards
        rewards = rewards_agents[agent_ind].clone()
        n = rewards.size(0)

        # Compute classification loss
        l1 = classification_loss(torch.stack(all_log_probs[agent_ind]), target_class.repeat(n))

        for i in range(n - 2, -1, -1):
            rewards[i] = rewards[i] + args.gamma * rewards[i + 1]

        l2 = -((rewards - rewards_mean[agent_ind][:n]) * log_probs[agent_ind]).sum()

        # Compute loss
        losses[agent_ind] = 10 * l1 + l2
        # Update the baseline
        rewards_mean[agent_ind][:n] = 0.9 * rewards_mean[agent_ind][:n] + 0.1 * rewards.detach()


if __name__ == '__main__':

    # create the required directories
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # instantiate agents and their corresponding traits
    agents, traits_agents = instantiate_agents()
    pickle.dump(traits_agents, traits_file)
    traits_file.close()
    # calculate total reputation by summing all the 0th elements of the reputation lists
    total_reputation = torch.tensor(0.0)
    for agent_id in range(args.num_agents):
        total_reputation += traits_agents[agent_id][2][0]
    # initialize optimizers
    optimizers = dict([(agent_id, optim.Adam(agents[agent_id].parameters(),
                                             lr=args.learning_rate)) for agent_id in range(args.num_agents)])
    # divide dataset into train and test sets
    train_set = utils.process_dataset()
    # classes in CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Initialize running mean of rewards
    rewards_mean = dict([(agent, torch.zeros(args.num_steps)) for agent in agents])
    running_rewards = 0.0
    num_consensus = 0  # number of times the community reaches consensus
    avg_steps, step = 0, 0  # time needed on as average to reach consensus

    for epoch in range(4):
        # start running episodes
        for episode_id in range(args.num_episodes):
            # pick up one sample from train dataset
            (img, target) = train_set[episode_id]
            img = img.unsqueeze(0)
            target = torch.tensor([target])

            # initialize / re-initialize all parameters
            traits = [traits_agents[agent_id].clone() for agent_id in range(args.num_agents)]
            agent_all_log_probs = dict([(agent_id, []) for agent_id in range(args.num_agents)])
            agent_log_probs = dict([(agent_id, []) for agent_id in range(args.num_agents)])
            agent_rewards = dict([(agent_id, []) for agent_id in range(args.num_agents)])
            msgs_input = dict([(agent_id, torch.zeros(args.num_agents - 1, args.msg_v_dim + args.msg_k_dim))
                               for agent_id in range(args.num_agents)])
            msgs_broadcast = dict([(agent_id, torch.zeros(args.num_agents - 1, args.msg_v_dim))
                                   for agent_id in range(args.num_agents)])
            agent_loss = torch.tensor(0.0)
            losses = dict()

            # reset agent hidden vectors of LSTM and traits to original values
            for agent_id in range(args.num_agents):
                agents[agent_id].reset_agent(traits[agent_id])

            # reputation = dict([(agent_id, torch.zeros(args.num_agents - 1)) for agent_id in range(args.num_agents)])
            if episode_id == 48000:
                resources_file.write('episode:' + str(episode_id) + '\n')
            # begin episode
            for step in range(args.num_steps):
                consensus = [0] * args.num_classes
                for agent_id in range(args.num_agents):

                    # agents get to see the cropped image for partial observability
                    cropped_image = utils.crop_image(img, str(agent_id))
                    traits[agent_id], action, predicted_probs, msgs_broadcast[agent_id], all_probs = \
                        agents[agent_id](cropped_image,
                                         msgs_input[agent_id])

                    agent_log_probs[agent_id].append(predicted_probs)
                    agent_all_log_probs[agent_id].append(all_probs.squeeze(0))

                    if episode_id == 48000:
                        resources_file.write('agent:' + str(agent_id) +
                                             ' resource allocation: ' + str(traits[agent_id][0]) + '\n')

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

                    # for neighbour in range(args.num_agents):
                    #     if neighbour != agent_id:
                    #         msgs_input[neighbour][index] = msgs_broadcast[index].clone()
                    #         # reputation[neighbour] += attention[index].detach().repeat(args.num_agents - 1)
                    #         index += 1

                # for agent_id in range(args.num_agents):
                #     traits[agent_id][2] = reputation[agent_id] / (args.num_agents - 1)
                # total_reputation += traits[agent_id][2][0]

                # prize = np.array(consensus)[target.item()]
                # for agent_id in range(args.num_agents):
                #     prize_share = (traits[agent_id][2][0].detach() / total_reputation) * prize
                #     agent_rewards[agent_id].append(prize_share)

                # define reward function
                if max(np.array(consensus)) > args.threshold and \
                        torch.tensor(consensus).argmax().unsqueeze(0) == target:
                    # final_pred = torch.tensor(consensus).argmax().unsqueeze(0)
                    # if final_pred == target:
                    # print('episode:', episode_id, 'steps taken:', step, '(reached consensus with correct prediction)')
                    prize = args.prize
                    num_consensus += 1

                    for agent_id in range(args.num_agents):
                        prize_share = (traits[agent_id][2][0].detach() / total_reputation) * prize
                        agent_rewards[agent_id].append(prize_share.item())
                    break
                # elif step == args.num_steps - 1:
                #     print('episode:', episode_id, '(did not reach consensus)')
                #     final_pred = torch.tensor(consensus).unsqueeze(0)
                #     for agent_id in range(args.num_agents):
                #         agent_rewards[agent_id].append(-args.penalize)
                else:
                    for agent_id in range(args.num_agents):
                        prize_share = (traits[agent_id][2][0].detach() / total_reputation) * (-1)
                        agent_rewards[agent_id].append(prize_share)

            agent_rewards = [torch.tensor(agent_rewards[agent_id]) for agent_id in range(args.num_agents)]
            print_rewards = [agent_rewards[agent_id].sum().item() for agent_id in range(args.num_agents)]
            agent_log_probs = [torch.cat([x.view(-1, 1) for x in agent_log_probs[agent_id]], dim=1).squeeze()
                               for agent_id in range(args.num_agents)]

            compute_loss(target, agent_all_log_probs, agent_rewards, agent_log_probs)

            for optim in optimizers.values():
                optim.zero_grad()

            for agent_id in range(args.num_agents):
                agent_loss += losses[agent_id].clone()
            agent_loss.backward()

            # for agent_id in range(args.num_agents):
            #     clip_grad_value_(agents[agent_id].parameters(), 1.0)

            for optim in optimizers.values():
                optim.step()

            # print mean reward of every 2000 steps
            running_rewards += np.array(print_rewards).sum()
            avg_steps += step
            if episode_id == 0 and epoch == 0:
                print('[{}, {}] rewards: {} step: {} (reached consensus: {} / 2000)'.format(epoch, episode_id,
                                                                                            -25.00, 25, num_consensus))
                log_file.write('Episode: ' + str(episode_id) + ' rewards: ' + str(-25.00) + ' steps: ' + str(25)
                               + ' consensus: ' + str(num_consensus) + '/2000 ' + "\n")
                running_rewards = 0.0
                num_consensus = 0

            elif episode_id % 2000 == 0:  # print every 2000 mini-batches
                print('[{}, {}] rewards: {} steps: {} (reached consensus: {} / 2000)'.format(epoch, episode_id,
                                                                                             running_rewards / 2000,
                                                                                             avg_steps / 2000,
                                                                                             num_consensus))
                log_file.write('Episode: ' + str(episode_id) + ' rewards: ' + str(running_rewards / 2000)
                               + ' steps: ' + str(avg_steps / 2000) + ' consensus: ' +
                               str(num_consensus) + '/2000 ' + "\n")
                running_rewards = 0.0
                num_consensus = 0
                avg_steps = 0

            log_file.flush()

            # save every 8000th model parameters
            if episode_id % 8000 == 0 and episode_id != 0:
                for agent_id in range(args.num_agents):
                    with open(os.path.join(model_path, str(episode_id) + '_' + str(agent_id)), 'wb') as f:
                        torch.save(agents[agent_id].state_dict(), f)
            # traits = [traits[agent_id].detach() for agent_id in range(args.num_agents)]

            gc.collect()

        resources_file.write('\n')

    log_file.close()
    resources_file.close()
