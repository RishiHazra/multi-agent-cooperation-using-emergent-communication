import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import utils
from arguments import Arguments

args = Arguments()


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#         self.log_softmax = nn.LogSoftmax(dim=-1)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.log_softmax(self.fc3(x))
#         return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 6, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 12, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.fc1 = nn.Linear(12 * 3 * 3, args.encoded_size)
        self.fc2 = nn.Linear(args.encoded_size, 10)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, cropped_img):
        x = self.layer(cropped_img).view(-1, 12 * 3 * 3)
        x = torch.relu(self.fc1(x))
        y = self.log_softmax(self.fc2(x))
        return y


def process_dataset():
    """
    :return: dataset pre-processing step for CIFAR-10
    """
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


def compute_loss(target_class, all_log_probs):
    classification_loss = nn.CrossEntropyLoss()
    l1 = classification_loss(all_log_probs, target_class)
    return l1


if __name__ == '__main__':
    models = dict([(agent_ind, Net()) for agent_ind in range(4)])
    optimizers = dict([(agent_id, optim.Adam(models[agent_id].parameters(),
                                             lr=args.learning_rate)) for agent_id in range(4)])
    # divide dataset into train and test sets
    train_set, test_set = process_dataset()
    # classes in CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    running_loss = 0.0
    num_correct = 0

    # start running episodes
    for _ in range(5):
        for episode_id in range(args.num_episodes):
            # pick up one sample from train dataset
            (img, target) = train_set[episode_id]
            img = img.unsqueeze(0)
            target = torch.tensor([target])
            predicted = torch.zeros(10)

            for agent_id in range(4):
                cropped_image = utils.crop_image(img, agent_id)

                predicted_log_probs = models[agent_id](cropped_image)
                predicted[predicted_log_probs.argmax().item()] += 1

                loss = compute_loss(target, predicted_log_probs)
                optimizers[agent_id].zero_grad()
                loss.backward()
                optimizers[agent_id].step()

            pred = predicted.argmax().item()
            if pred > 2 and pred == target.item():
                num_correct += 1
            # running_loss += loss.item()
            if episode_id % 2000 == 0:  # print every 2000 mini-batches
                print('[{}] total correct: {} / 2000'.format(episode_id, num_correct))
                num_correct = 0.0
