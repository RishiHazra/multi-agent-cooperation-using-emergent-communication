import torch
import torch.nn as nn
from torch.distributions import Categorical

from arguments import Arguments
from agent_model import ImgEncoder, Receiver, Sender, Decoder

args = Arguments()


class Agent(nn.Module):
    def __init__(self, traits):
        super(Agent, self).__init__()
        self.img_encoder = ImgEncoder()
        self.receiver = Receiver()
        self.sender = Sender()
        self.decoder = Decoder()
        self.softmax = nn.Softmax(dim=0)
        self.traits = traits
        self.resource_division = nn.Linear(args.num_agents-1, args.num_agents-1)
        self.msgs_hidden = torch.zeros(1, args.msg_v_dim)

    def reset_agent(self, traits):
        self.msgs_hidden = torch.zeros(1, args.msg_v_dim)
        self.traits = traits

    def forward(self, image, msgs_input):
        img_enc = self.img_encoder(image)
        pooled, _ = self.receiver(msgs_input, self.traits[1].clone())
        self.msgs_hidden, msgs_broadcast = \
            self.sender(pooled, self.msgs_hidden, img_enc, self.traits[0].clone())

        log_probs = self.decoder(img_enc, pooled)
        probs = torch.exp(log_probs)
        action = Categorical(probs=probs).sample()
        log_probs = log_probs.gather(1, action.view(-1, 1)).view(-1)
        self.traits[0] = self.softmax(self.resource_division(self.traits[0].clone().unsqueeze(0)))[0]
        return self.traits, action, log_probs, msgs_broadcast
