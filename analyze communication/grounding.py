"""
visualize word embeddings
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


def fashion_scatter(x, colors):
    # choose a color palette with seaborn
    palette = np.array(sns.color_palette("hls", 5))
    plt.figure(figsize=(5, 5))
    ax = plt.subplot(aspect='equal')
    map_color = colors
    ax.scatter(x[:, 0], x[:, 1], c=palette[map_color])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    # txts = []
    # for ind in range(args.num_classes):
    #     # Position of each label at median of data points.
    #
    #     xtext, ytext = np.median(x[np.array(map_color) == ind, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(ind), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)
    # plt.title('(1,2)')
    plt.show()
    # f.savefig('word_embed' + str(ind) + '.png')


def SVD(pmi_mat):
    uu, ss, vv = linalg.svds(pmi_mat, 2)
    unorm = uu / np.sqrt(np.sum(uu * uu, axis=1, keepdims=True))
    # vnorm = vv / np.sqrt(np.sum(vv * vv, axis=0, keepdims=True))
    word_vecs = vv.T
    # word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs * word_vecs, axis=1, keepdims=True))
    return unorm, word_vecs


def word_embed():
    palette = np.array(sns.color_palette("hls", 10))
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(aspect='equal')

    for label_id in range(args.num_classes):
        Y = TSNE(random_state=RS).fit_transform(X[str(label_id)])
        ax.scatter(Y[:, 0], Y[:, 1], c=palette[label_id])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.show()


if __name__ == '__main__':
    # search_path = input("Enter directory path to search : ")
    search_path = '00000'
    episodes = np.arange(0, 1000, 1)

    X = defaultdict(list)  # [num_classes, num_msgs, msg_dim]
    label = defaultdict(list)
    f1 = None

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
                    label[fname.split('_')[3]] = [fname.split('_')[-2]]

    for file in all_files:
        f1 = open(file)

        lines = f1.readlines()

        if len(lines) < args.num_steps:  # only episodes where correct prediction was made
            for line in lines[-1:]:  # only consider the last message passing step (for each agent)
                line = line.strip('\n').split('\t')
                messages = np.split(np.array(line), args.num_agents - 1)

                if len(X[file.split('_')[5]]) < 200:  # consider atmost 200 messages for each label
                    X[file.split('_')[5]] += [messages[0].tolist()[5:]]  # msg_v (msg_k is not considered)

    word_embed()
