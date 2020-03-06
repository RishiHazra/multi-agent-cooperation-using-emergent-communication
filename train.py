import gc
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import clip_grad_value_
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
    train_sets = torchvision.datasets.CIFAR10(root='./cifar-10', train=True, download=False, transform=transform_train)
    test_sets = torchvision.datasets.CIFAR10(root='./cifar-10', train=False, download=False, transform=transform_test)
    return train_sets, test_sets


def crop_image(image, agent):
    locs = {
        '0': (0, 16, 0, 16),
        '1': (0, 16, 16, 32),
        '2': (16, 32, 0, 16),
        '3': (16, 32, 16, 32),
        # '4': (16, 32, 0, 16),
        # '5': (16, 32, 0, 16),
        # '6': (16, 32, 16, 32),
        # '7': (16, 32, 16, 32)
    }
    x1, x2, y1, y2 = locs[agent]
    return image[:, :3, x1:x2, y1:y2]


def instantiate_agents():
    agent_traits = dict((agent_ind, list()) for agent_ind in range(args.num_agents))
    for agent_ind in range(args.num_agents):
        agent_traits[agent_ind] = torch.softmax(torch.ones(args.num_agents - 1, requires_grad=False).unsqueeze(0),
                                                dim=-1)
        agent_traits[agent_ind] = torch.cat((agent_traits[agent_ind], torch.rand(1, dtype=torch.float32,
                                                                                 requires_grad=False).repeat(
            args.num_agents - 1).unsqueeze(0)))
        agent_traits[agent_ind] = torch.cat((agent_traits[agent_ind],
                                             torch.randint(1, 2, (1,), dtype=torch.float32, requires_grad=False).repeat(
                                                 args.num_agents - 1).unsqueeze(0)))
        print(agent_traits[agent_ind][2])
    agent_nws = dict([(agent_ind, Agent(agent_traits[agent_ind])) for agent_ind in range(args.num_agents)])
    return agent_nws, agent_traits


def compute_loss(rewards_agents, log_probs):
    # Compute loss for each agent
    for agent_ind in range(args.num_agents):

        # Compute cumulative rewards
        rewards = rewards_agents[agent_ind].clone()
        n = rewards.size(0)
        for i in range(n - 2, -1, -1):
            rewards[i] = rewards[i] + args.gamma * rewards[i + 1]

        # Compute loss
        losses[agent_ind] = -((rewards - rewards.mean()) * log_probs[agent_ind]).mean()


if __name__ == '__main__':
    agents, traits = instantiate_agents()
    # print(traits)
    total_reputation = torch.tensor(0.0)
    for agent_id in range(args.num_agents):
        total_reputation += traits[agent_id][2][0]
    print(total_reputation)
    optimizers = dict([(agent_id, Adam(agents[agent_id].parameters(),
                                       lr=args.learning_rate)) for agent_id in range(args.num_agents)])
    train_set, test_set = process_dataset()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for episode_id in range(args.num_episodes):
        (img, target) = train_set[episode_id]
        img = img.unsqueeze(0)
        target = torch.tensor([target])
        agent_log_probs = dict([(agent_id, []) for agent_id in range(args.num_agents)])
        agent_rewards = dict([(agent_id, []) for agent_id in range(args.num_agents)])
        msgs_input = dict([(agent_id, torch.zeros(args.num_agents - 1, args.msg_k_dim + args.msg_v_dim))
                           for agent_id in range(args.num_agents)])
        agent_loss = torch.tensor(0.0)
        losses = dict()

        for agent_id in range(args.num_agents):
            agents[agent_id].reset_agent(traits[agent_id])
        # reputation = dict([(agent_id, torch.zeros(args.num_agents - 1)) for agent_id in range(args.num_agents)])

        for step in range(args.num_steps):
            consensus = [0] * args.num_classes
            for agent_id in range(args.num_agents):
                cropped_image = crop_image(img, str(agent_id))
                traits[agent_id], action, predicted_probs, msgs_broadcast = \
                    agents[agent_id](cropped_image,
                                     msgs_input[agent_id])

                agent_log_probs[agent_id].append(predicted_probs)

                index = 0
                for neighbour in range(args.num_agents):
                    if neighbour != agent_id:
                        msgs_input[neighbour][index] = msgs_broadcast[index].clone()
                        # reputation[neighbour] += attention[index].detach().repeat(args.num_agents - 1)
                        index += 1

                # print(step, agent_id, traits[agent_id][2][0].item())
                if step != 0:
                    consensus[action.item()] += traits[agent_id][2][0].item()
                    # consensus[action.item()] += 1
                    # print('step:', step, ' agent:', agent_id, ' consensus:', consensus, ' target:', target)

            # for agent_id in range(args.num_agents):
            #     traits[agent_id][2] = reputation[agent_id] / (args.num_agents - 1)
            # total_reputation += traits[agent_id][2][0]

            # prize = np.array(consensus)[target.item()]
            # for agent_id in range(args.num_agents):
            #     prize_share = (traits[agent_id][2][0].detach() / total_reputation) * prize
            #     agent_rewards[agent_id].append(prize_share)

            if max(np.array(consensus)) > args.threshold and torch.tensor(consensus).argmax().unsqueeze(0) == target:
                # final_pred = torch.tensor(consensus).argmax().unsqueeze(0)
                # if final_pred == target:
                print('episode:', episode_id, 'steps taken:', step, '(reached consensus with correct prediction)')
                prize = args.prize
                # else:
                #     print('episode:', episode_id, 'steps taken:', step, '(reached consensus with wrong prediction)')
                #     prize = -args.num_agents
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
                    prize_share = (traits[agent_id][2][0].detach() / total_reputation) * (-0.5)
                    agent_rewards[agent_id].append(prize_share)

        agent_rewards = [torch.tensor(agent_rewards[agent_id]) for agent_id in range(args.num_agents)]
        print_rewards = [agent_rewards[agent_id].sum().item() for agent_id in range(args.num_agents)]
        agent_log_probs = [torch.cat([x.view(-1, 1) for x in agent_log_probs[agent_id]], dim=1).squeeze()
                           for agent_id in range(args.num_agents)]

        compute_loss(agent_rewards, agent_log_probs)

        for optim in optimizers.values():
            optim.zero_grad()

        for agent_id in range(args.num_agents):
            # print('agent:', agent_id, 'losses:', losses[agent_id])
            agent_loss += losses[agent_id].clone()
        print('target:', target.item(), '| total rewards:', np.array(print_rewards).sum(), '| consensus:', consensus, '\n')
        agent_loss.backward()

        for agent_id in range(args.num_agents):
            clip_grad_value_(agents[agent_id].parameters(), 1.0)

        for optim in optimizers.values():
            optim.step()

        traits = [traits[agent_id].detach() for agent_id in range(args.num_agents)]

        gc.collect()
