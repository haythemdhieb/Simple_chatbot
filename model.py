import torch
import torch.nn as nn
import sys

sys.path.append("/home/haythem/Desktop/Work/training/chatbot/")
from utils import preproces_text


class chatbotmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(chatbotmodel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.layer1(x)
        output = self.relu(output)
        output = self.layer2(output)
        output = self.relu(output)
        output = self.layer3(output)
        output = self.relu(output)
        return output
