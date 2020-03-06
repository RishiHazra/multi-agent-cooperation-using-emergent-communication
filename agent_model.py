import torch
import torch.nn as nn

from arguments import Arguments

args = Arguments()


# class ImgEncoder(nn.Module):
#     def __init__(self):
#         super(ImgEncoder, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(3, 6, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(6, 16, 5),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2))
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, args.encoded_size)
#
#     def forward(self, image):
#         x = self.layers(image).view(-1, 16 * 5 * 5)  # [b, 16, 5, 5]
#         x = torch.relu(self.fc1(x))  # [b, 120]
#         x = torch.relu(self.fc2(x))  # [b, 84]
#         return x

class ImgEncoder(nn.Module):
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


class Receiver(nn.Module):
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
    def __init__(self):
        super(Sender, self).__init__()
        self.gru = nn.GRUCell(args.msg_v_dim, args.msg_v_dim)
        self.w_enc = nn.Linear(args.encoded_size, args.msg_v_dim)
        self.w_msg = nn.Linear(args.msg_v_dim, args.msg_v_dim)
        self.layer_resources = nn.Linear(1, args.msg_k_dim)

    def forward(self, pooled, msg_hidden_old, img_enc, resources):
        msg_hidden_new = self.gru(pooled.unsqueeze(0), msg_hidden_old)
        msg_v = self.w_enc(img_enc) + self.w_msg(msg_hidden_new)
        msg_k = self.layer_resources(resources.unsqueeze(1))
        # different messages for different agents
        msgs_broadcast = torch.cat((msg_k, msg_v[0].clone().repeat(args.num_agents-1, 1)), dim=-1)  # [num_agents-1,10]
        return msg_hidden_new, msgs_broadcast


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.w_img = nn.Linear(args.encoded_size, args.num_classes)
        self.w_msg = nn.Linear(args.msg_v_dim, args.num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, encoder_out, msg_pooled):
        y = self.w_img(encoder_out) + self.w_msg(msg_pooled.unsqueeze(0))
        # y = self.w_msg(msg_pooled.unsqueeze(0))
        y = self.log_softmax(y)
        # print(y)
        return y
