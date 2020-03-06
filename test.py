import gc
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms

from torch.optim import Adam

from agent import Agent
from arguments import Arguments

args = Arguments()


def process_dataset():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_sets = torchvision.datasets.CIFAR10(root='./cifar-10', train=True, download=True, transform=transform_train)
    test_sets = torchvision.datasets.CIFAR10(root='./cifar-10', train=False, download=True, transform=transform_test)
    return train_sets, test_sets


def instantiate_agents(evaluate):
    agent_traits = dict((agent_id, list())
                        for agent_id in range(args.num_agents))
    for agent_id in range(args.num_agents):
        agent_traits[agent_id] = torch.softmax(torch.tensor(torch.ones(args.num_agents - 1)).unsqueeze(0), dim=-1)
        agent_traits[agent_id] = torch.cat((agent_traits[agent_id],
                                            torch.tensor(np.random.random(args.num_agents - 1),
                                                         dtype=torch.float32).unsqueeze(0)))
        agent_traits[agent_id] = torch.cat((agent_traits[agent_id],
                                            torch.tensor(torch.ones(args.num_agents - 1)).unsqueeze(0)))

    agent_nws = dict([(agent_id, Agent(agent_traits[agent_id]))
                      for agent_id in range(args.num_agents)])
    return agent_nws, agent_traits


def reward_function():
    return 0


if __name__ == '__main__':
    agents, traits = instantiate_agents()
    optimizers = dict([(agent_id, Adam(agents[agent_id].parameters(),
                                       lr=args.learning_rate)) for agent_id in range(args.num_agents)])
    total_resources = dict([(agent_id, sum(traits[agent_id][0])) for agent_id in range(args.num_agents)])

    train_set, test_set = process_dataset()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for episode_id in range(args.num_episodes):
        net_reward = 0

        consensus = dict([(label, 0) for label in range(args.num_classes)])
        agent_log_probs = dict([(agent, []) for agent in agents])
        msgs_hidden_old = dict([(agent_id, torch.zeros(args.msg_v_dim)) for agent_id in range(args.num_agents)])
        msgs_input = dict([(agent_id, torch.zeros(args.num_agents - 1, args.msg_k_dim + args.msg_v_dim))
                           for agent_id in range(args.num_agents)])
        img = dict([(agent_id, torch.zeros(32, 32)) for agent_id in range(args.num_agents)])
        reputation = dict([(agent_id, torch.zeros(args.num_agents - 1)) for agent_id in range(args.num_agents)])

        for step in range(args.num_steps):
            for agent_id in range(args.num_agents):
                predicted_probs, msgs_broadcast, msg_hidden_new, attention = \
                                                                            agents[agent_id](img[agent_id],
                                                                                             msgs_input[agent_id],
                                                                                             msgs_hidden_old[agent_id])
                action = predicted_probs.max(1)[1]
                consensus[action] += reputation[agent_id][0]
                agent_log_probs[agent_id].append(torch.log(predicted_probs.max(1)[0]))

                index = 0
                for neighbour in range(args.num_agents):
                    if neighbour != agent_id:
                        msgs_input[neighbour][index] = msgs_broadcast[index]
                        if index == 0:
                            reputation[agent_id] = attention.repeat(args.num_agents - 1)
                        else:
                            reputation[neighbour] = torch.sum(reputation[agent_id],
                                                              attention[index].repeat(args.num_agents - 1))
                        index += 1

                msgs_hidden_old[agent_id] = msg_hidden_new

            for agent_id in range(args.num_agents):
                traits[agent_id][2] = reputation[agent_id] / (args.num_agents - 1)

            if consensus.max(1)[0] > args.threshold:
                final_pred = consensus.max(1)[1]
                break
            else:
                net_reward -= 1

        gc.collect()

# TODO : add model.eval()
