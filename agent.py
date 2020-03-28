from agent_model import *


def weights_init(module):
    """
    initializer of Linear weights
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        # nn.init.constant_(module.bias, 0)


class Agent(nn.Module):
    def __init__(self, traits):
        super(Agent, self).__init__()
        self.img_encoder = ImgEncoder()
        self.text_encoder = CharEncoder()
        self.receiver = Receiver()
        self.sender = Sender()
        self.decoder = Decoder()
        self.softmax = nn.Softmax(dim=-1)
        self.traits = traits
        self.resource_division = nn.Linear(args.num_agents - 1, args.num_agents - 1)
        self.msgs_hidden = torch.zeros(1, args.msg_v_dim)
        if args.flag == 1:
            self.char_hidden = torch.zeros(1, args.encoded_size)
        self.apply(weights_init)

    def reset_agent(self, traits):
        self.msgs_hidden = torch.zeros(1, args.msg_v_dim)
        if args.flag == 1:
            self.char_hidden = torch.zeros(1, args.encoded_size)
        self.traits = traits

    def forward(self, observation, msgs_input):
        """
        :param observation: partial observation of agent
        :param msgs_input: input messages from other agents
        :return: modified traits (resource division), action taken,
                log_probabilities of all actions, broadcast messages
        """
        # encode observation
        if args.flag == 0:  # encode images
            enc = self.img_encoder(observation)
        else:  # encode text
            self.char_hidden = self.text_encoder(observation, self.char_hidden)

        # message pooling
        pooled, _ = self.receiver(msgs_input, self.traits[1].clone())

        # resource division
        self.traits[0] = self.softmax(self.resource_division(self.traits[0].clone().unsqueeze(0)))[0]

        # broadcast messages
        if args.flag == 0:
            self.msgs_hidden, msgs_broadcast = \
                self.sender(pooled, self.msgs_hidden, enc, self.traits[0].clone())
            # take actions using Straight-Through Gumbel Softmax
            action, log_probs, all_classes_log_probs = self.decoder(enc, pooled)
        else:
            self.msgs_hidden, msgs_broadcast = \
                self.sender(pooled, self.msgs_hidden, self.char_hidden, self.traits[0].clone())
            # take actions using Straight-Through Gumbel Softmax
            action, log_probs, all_classes_log_probs = self.decoder(self.char_hidden, pooled)

        return self.traits, action, log_probs, msgs_broadcast, all_classes_log_probs
