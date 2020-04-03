import numpy as np
import torchvision
import torchvision.transforms as transforms


def process_dataset(evaluate=False):
    """
    :return: dataset pre-processing step for CIFAR-10
    """
    if not evaluate:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_sets = torchvision.datasets.CIFAR10(root='./cifar-10', train=True, download=False,
                                                  transform=transform_train)
        return train_sets
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_sets = torchvision.datasets.CIFAR10(root='./cifar-10', train=False, download=False,
                                                 transform=transform_test)
        return test_sets


def crop_image(image, agent):
    """
    :param image: input sample in an episode
    :param agent: agent id
    :return: cropped image for agent
    """
    locs = {
        '0': (0, 16, 0, 16),
        '1': (0, 16, 16, 32),
        '2': (16, 32, 0, 16),
        '3': (16, 32, 16, 32)
    }
    index = str(int(agent) % 4)
    x1, x2, y1, y2 = locs[index]
    return image[:, :3, x1:x2, y1:y2]


def index_chars(all_labels):
    """
    :param all_labels: all classes in a list
    :return: indexing dict
    """
    unique_chars = list(set(list(''.join(all_labels))))
    unique_chars += '$'  # special character
    char_map = dict([(char, unique_chars.index(char)) for char in unique_chars])
    return char_map


def generate_reputation(to_go, sum_repo, upper_limit):
    """
    :param to_go: total number of agents
    :param sum_repo: sum of all reputations (constant)
    :param upper_limit: highest reputation - 1
    :return: list of reputation of all_agents
    """
    repo = []
    while sum_repo != 0 and to_go > 0:
        diff = sum_repo - to_go
        if diff > 1 and to_go > 1:
            new = np.random.randint(1, upper_limit, (1,))[0]
            repo += [new]  # append generated reputation to the list
            sum_repo -= new
            to_go -= 1
        elif to_go == 1 and diff > upper_limit - 1:
            repo += [diff]
            break
        else:
            repo += [1] * to_go  # once sum if equal to the number of agents, remaining all get one
            break
    np.random.shuffle(repo)
    return repo
