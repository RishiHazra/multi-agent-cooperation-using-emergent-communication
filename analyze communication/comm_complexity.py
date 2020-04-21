"""
Parzen-window estimation for communication complexity
"""
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from arguments import Arguments

args = Arguments()


def l2dist(a, b):
    return np.linalg.norm(a - b)


def parzen_window_est(x_samples, center, h=1):
    """
    Implementation of the Parzen-window estimation for hypercubes.

    Keyword arguments:
        x_samples: A 'n x d'-dimensional numpy array, where each sample
            is stored in a separate row.
        h: The length of the hypercube.
        center: The coordinate center of the hypercube

    Returns the probability density for observing k samples inside the hypercube.

    """
    dimensions = x_samples.shape[1]
    k = 0
    for x in x_samples:
        is_inside = 1
        for axis, center_point in zip(x, center):
            if np.abs(float(axis) - float(center_point)) > (h / 2):
                is_inside = 0
        k += is_inside
    return (k / len(x_samples)) / (h ** dimensions)


def calculate_entropy(samples_X):
    entropy = 0
    print('calculating entropy of the system for {} points ...'.format(len(samples_X)))
    for sample in tqdm(range(len(samples_X))):
        prob = parzen_window_est(samples_X, center=samples_X[sample])
        entropy += np.log(prob) * prob

    print('Entropy of the system:', -entropy)
    return -entropy


if __name__ == '__main__':
    # search_path = input("Enter directory path to search : ")
    # search_path = '../00000'  # 4-agents
    # repo = [1, 1, 1, 1, 2, 1, 2, 1, 2, 2]  # 10 agents
    # repo = [1, 1, 3, 2, 1, 1, 1, 1, 2, 1]  # limited labs
    # repo = [1, 1, 3, 1, 2, 1, 1, 1, 1, 2]  # 10 agents bar
    # search_path = '00000' # 10-agents restricted comm; limited labs+ repo: [3, 2, 2, 1, 1, 1, 1, 1, 1]
    search_path = '0000010agents'
    neighbors = {0: [1, 2], 1: [0, 3, 4, 5, 6, 7, 8, 9], 2: [0, 3, 4, 5, 6, 7, 8, 9], 3: [1, 2], 4: [1, 2],
                 5: [1, 2], 6: [1, 2], 7: [1, 2], 8: [1, 2], 9: [1, 2]}

    episodes = np.arange(0, 445, 1)

    X = defaultdict(list)  # [num_classes, num_msgs, msg_dim]
    f1 = None

    files_to_search = []
    for agent_id in range(args.num_agents):
        files_to_search += ['Im_msg_' + str(agent_id) + '_EP_']

    all_files = []

    print('\n---------------------------------\n')
    print('fetching files for {} episodes...'.format(len(episodes)))
    for episode in tqdm(episodes):
        files1 = list(map(lambda x: x + str(episode) + '_', files_to_search))

        """
        file format: msg_0_EP_0_lab_0_48000
        msg_0: agent 0
        EP_0: episode 0
        lab_0: label 0 (label from the corresponding episode)
        48000: policy number
        """
        for file in files1:
            for fname in os.listdir(path=search_path):
                if fname.startswith(file):
                    all_files += [search_path + '/' + fname + '/tensors.tsv']

    for file in all_files:
        f1 = open(file)
        lines = f1.readlines()
        # agent_index = int(file.split('_')[2])

        for line in lines[-1:]:  # only consider the last message passing step (for each agent)
            line = line.strip('\n').split('\t')
            # num_neighbors = len(neighbors[agent_index])
            # messages = np.split(np.array(line), 10 - 1)
            messages = np.split(np.array(line), 9)

            for message in messages:
                X[file.split('_')[2]] += [message]  # msg_v (msg_k is not considered) for each agent

        # # if len(lines) < args.num_steps:  # only episodes where correct prediction was made
        # for line in lines[-1:]:  # only consider the last message passing step (for each agent)
        #     line = line.strip('\n').split('\t')
        #     messages = np.split(np.array(line), 4 - 1)
        #
        #     X += [messages[0][5:]]  # msg_v (msg_k is not considered)

    avg = 0
    for agent, values in X.items():  # calculate entropy on a per agent basis
        print('\n---------------------------------\n')
        print('agent:', agent)
        avg += calculate_entropy(np.array(values))
    print('average entropy: ', avg / len(X.items()))
