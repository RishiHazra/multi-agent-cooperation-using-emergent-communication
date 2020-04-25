"""
visualize word embedding signatures (1 agent, 1 label, different signatures for neighbours)
"""
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import linalg
from sklearn.manifold import TSNE

from arguments import Arguments

RS = 123

args = Arguments()
sns.set_palette('hls')


def word_embed(neighbours):
    palette = np.array(sns.color_palette("hls", len(neighbours)))
    plt.figure(figsize=(4, 4))
    ax = plt.subplot(aspect='equal')

    for ind, label_id in enumerate(neighbours):
        Y = TSNE(random_state=RS).fit_transform(X[label_id])
        ax.scatter(Y[:, 0], Y[:, 1], c=palette[ind])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.show()


if __name__ == '__main__':
    # search_path = input("Enter directory path to search : ")   
    # neighbors = {0: [4, 5, 6, 7, 8, 9], 1: [4, 5, 6, 7, 8, 9], 2: [4, 5, 6, 7, 8, 9], 3: [4, 5, 6, 7, 8, 9],
    #              4: [0, 1, 2, 3], 5: [0, 1, 2, 3], 6: [0, 1, 2, 3], 7: [0, 1, 2, 3], 8: [0, 1, 2, 3], 9: [0, 1, 2, 3]}

    neighbors = {0: [1, 2], 1: [0, 3, 4, 5, 6, 7, 8, 9], 2: [0, 3, 4, 5, 6, 7, 8, 9], 3: [1, 2], 4: [1, 2],
                 5: [1, 2], 6: [1, 2], 7: [1, 2], 8: [1, 2], 9: [1, 2]}

    episodes = np.arange(0, 1000, 1)

    X = defaultdict(list)  # [num_classes, num_msgs, msg_dim]
    label = defaultdict(list)
    f1 = None

    analyze_agent = 1
    analyze_label = 0

    files_to_search = []
    for agent_id in range(args.num_agents):
        files_to_search += ['Im_msg_' + str(agent_id) + '_EP_']

    all_files = []

    for episode in episodes:
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
                    # label[fname.split('_')[3]] = [fname.split('_')[-2]]

    for file in all_files:
        f1 = open(file)
        lines = f1.readlines()

        agent_index = int(file.split('_')[2])
        label_index = int(file.split('_')[6])

        if agent_index == analyze_agent and label_index == analyze_label:
            num_neighbors = len(neighbors[agent_index])

            if len(lines) < args.num_steps:  # only episodes where correct prediction was made
                for line in lines[-1:]:  # only consider the last message passing step (for each agent)
                    line = line.strip('\n').split('\t')
                    messages = np.split(np.array(line), num_neighbors)

                    for index, message in enumerate(messages):
                        adjacent = neighbors[agent_index][index]

                        X[adjacent] += [message[:5]]
                        # X[file.split('_')[2]] += [message]  # msg_k (msg_v is not considered) for each agent
                        
    word_embed(neighbors[analyze_agent])
