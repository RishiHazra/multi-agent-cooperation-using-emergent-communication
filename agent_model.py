import torch
import torch.nn as nn

from arguments import Arguments

args = Arguments()


class ImgEncoder(nn.Module):
    """
    encodes the partial observations
    :param: cropped_image: partial image observations
    :returns: x: encoded image observations
    """

    def __init__(self):
        super(ImgEncoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 6, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 12, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.fc = nn.Linear(12 * 3 * 3, args.encoded_size)

    def forward(self, cropped_image):
        x = self.layer(cropped_image).view(-1, 12 * 3 * 3)
        x = torch.relu(self.fc(x))
        return x


class CharEncoder(nn.Module):
    """
    encodes characters for Scramble
    :param: char: indexed character (total indices: num unique characters)
            char_hidden_old: old hidden state of gru
    :return: char_hidden_new: new hidden state of gru
    """

    def __init__(self):
        super(CharEncoder, self).__init__()
        self.gru = nn.GRUCell(1, args.text_hidden)

    def forward(self, char, char_hidden_old):
        char_hidden_new = self.gru(char, char_hidden_old)
        return char_hidden_new


class Receiver(nn.Module):
    """
    Receives the input messages from all other agents
    and attends to them based on their key and query
    vectors. The query vectors are functions of intolerance.
    :param: msgs_input: all input messages [key, value]
            intolerance: intolerance parameter
    :return: pooled: pooled messages
             attention: attention weights
    """

    def __init__(self):
        super(Receiver, self).__init__()
        self.layer_query = nn.Linear(1, args.msg_k_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, msgs_input, intolerance):
        # [num_agents, 5],[num_agents, 5]
        msg_k, msg_v = msgs_input[:, :args.msg_k_dim].clone(), msgs_input[:, args.msg_k_dim:].clone()
        query = self.layer_query(intolerance[0].clone().view(-1))
        attention = self.softmax(msg_k.matmul(query))
        pooled = attention.matmul(msg_v)
        return pooled, attention


class Sender(nn.Module):
    """
    Broadcasts different messages to different agents.
    Keys are functions of resources parameter.
    :param: pooled: pooled messages from Receiver
            msg_hidden_old: hidden state of gru-cell
            img_enc: output of ImgEncoder used for calculating msg_v
            resources: resources at disposal
    :return: msgs_hidden_new: new hidden state of gru-cell
             msgs_broadcast: broadcasts messages to all agents
    """

    def __init__(self, neighbors):
        super(Sender, self).__init__()
        self.num_neighbors = len(neighbors)
        self.gru = nn.GRUCell(args.msg_v_dim, args.msg_v_dim)
        self.w_enc = nn.Linear(args.encoded_size, args.msg_v_dim)
        self.w_msg = nn.Linear(args.msg_v_dim, args.msg_v_dim)
        self.layer_resources = nn.Linear(1, args.msg_k_dim)

    def forward(self, pooled, msg_hidden_old, img_enc, resources):
        msg_hidden_new = self.gru(pooled.unsqueeze(0), msg_hidden_old)
        msg_v = self.w_enc(img_enc) + self.w_msg(msg_hidden_new)
        msg_k = self.layer_resources(resources.unsqueeze(1))
        # different messages for different agents # [num_agents-1,10]
        msgs_broadcast = torch.cat((msg_k, msg_v[0].clone().repeat(self.num_neighbors, 1)), dim=-1)
        return msg_hidden_new, msgs_broadcast


class Decoder(nn.Module):
    """
    Takes action in a differentiable environment (using
    a Straight Through version of Gumbel-Softmax sampling).
    Decodes the pooled messages and the image encodings
    :param: encoder_out: output of ImgEncoder
            msg_pooled: pooled messages from Receiver
    :return: action: output of Straight-through Gumbel Softmax
             log_probs: log probability of all actions
    """

    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device
        self.w_img = nn.Linear(args.encoded_size, args.num_classes)
        self.w_msg = nn.Linear(args.msg_v_dim, args.num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def sample_gumble(shape, eps=1e-20):
        u = torch.rand(shape)
        return -torch.log(-torch.log(u + eps) + eps)

    def gumble_softmax_sample(self, log_logits, temperature):
        y = log_logits + self.sample_gumble(log_logits.size()).to(self.device)
        return self.softmax(y / temperature)

    def forward(self, encoder_out, msg_pooled):
        y = self.w_img(encoder_out) + self.w_msg(msg_pooled.unsqueeze(0))
        y_log_probs = self.log_softmax(y)
        y_gumbel = self.gumble_softmax_sample(y_log_probs, args.temp)

        y_gumbel, ind = y_gumbel.max(dim=-1)
        y_hard = ind.float()
        action = (y_hard.to(self.device) - y_gumbel).detach() + y_gumbel
        return action.long(), y_log_probs.gather(1, action.long().view(-1, 1)).view(-1), y_log_probs
