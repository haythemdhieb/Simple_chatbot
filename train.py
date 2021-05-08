import sys

sys.path.append("/home/haythem/Desktop/Work/training/chatbot/")
from dataloader import chatbotdata
from model import chatbotmodel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch

hidden_size = 12
output_size = 7  # number of classes
input_size = len(chatbotdata().xdata[1])
batch_size = 8
learning_rate = 0.001
# loading the data
Chatbot_data = DataLoader(chatbotdata(), batch_size, shuffle=True, num_workers=2)
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
model = chatbotmodel(input_size, hidden_size, output_size).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 300
for epoch in range(num_epochs):
    for words, labels in Chatbot_data:
        words = words.to(device)
        labels = labels.to(device)
        outputs = model.forward(words)
        loss = loss_function(outputs, labels)
        # backward movement
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch+1}/{num_epochs},loss={loss.item():.4f}")
print(f"final loss,loss={loss.item():.4f}")
