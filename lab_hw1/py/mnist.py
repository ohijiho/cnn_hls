#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import debug

import sys, os.path
os.chdir(os.path.dirname(sys.argv[0]))

input_size = 784
# hidden_size = 500
output_size = 10

num_classes = 10
num_epochs = 5

batch_size = 100
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            download=True,
                            transform=transforms.Compose([transforms.ToTensor(),
                                                         # transforms.Normalize((0.1307,),(0.3081,))
                                                          ])
                           )

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                          # transforms.Normalize((0.1307,),(0.3081,))
                                                         ])
                           )

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

from functools import reduce
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # self.fc1 = nn.Linear(784,500)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(500,10)
###############################################################
### Your codes here ###########################################
###############################################################
        
        def ip(image_size, in_features, out_features):
            return nn.Linear(
                image_size[0] * image_size[1] * in_features,
                out_features)
        
        chain = [
            (nn.Conv2d(1, 5, (5, 5)), "conv1"),
            (nn.MaxPool2d((2, 2)),),
            (debug.Conv2d_dump(5, 5, (5, 5)), "conv2"),
            (nn.MaxPool2d((2, 2)),),
            (lambda x: x.view(-1, 80),),
            (ip((4, 4), 5, 40), "ip1"),
            (nn.Tanh(),),
            (ip((1, 1), 40, 10), "ip2"),
        ]

        # chain = [
        #     (nn.Conv2d(1, 5, (5, 5)), "conv1"),
        #     # (lambda x: x.view(-1, 1 * 28 * 28),),
        #     # (ip((28, 28), 1, 5 * 24 * 24), "conv1"),
        #     # (lambda x: x.view(-1, 5, 24, 24),),
        #     (nn.MaxPool2d((2, 2)),),
        #     (lambda x: x.view(-1, 5 * 12 * 12),),
        #     (ip((12, 12), 5, 40), "ip1"),
        #     (nn.Tanh(),),
        #     (ip((1, 1), 40, 10), "ip2"),
        # ]

        for i, u in enumerate(chain):
            f = u[0]
            setattr(self, f"param{i}", f)

        self.chain = chain

    def forward(self, x):
      # x = self.fc1(x)
      # x = self.relu1(x)
      # out = self.fc2(x)


###############################################################
### Your codes here ###########################################
###############################################################
        
        return reduce(lambda t, u: u[0](t), self.chain, x)

    def set_debug(self, b):
        for u in self.chain:
            f = u[0]
            if hasattr(f, "set_debug"):
                f.set_debug(b)

net = MLP()
net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # inputs = inputs.view(-1,784)
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1)%100 == 0:
            print("Epoch [%d/%d], Step [%d/%d], Loss: %.4f"
                  %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data))
            
correct = 0
total = 0

for inputs, labels in test_loader:
    # inputs = inputs.view(-1,784)
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    outputs = net(inputs)
    
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    
correct = torch.tensor(correct, dtype=torch.float32, device=device)
total = torch.tensor(total, dtype=torch.float32, device=device)

print("Accuacy of the network on the 10,000 test images:%.4f %%" % (100.0*correct/total))

torch.save(net.state_dict(), 'model.pkl')

# test_set.h and label.h test

def parse_list(s):
    code = f"[{s[s.index('{') + 1 : s.rindex('}')]}]".replace('\n', '')
    return eval(code)

with open("../data/test_set.h") as fts, \
        open("../data/label.h") as flb:
    lts = parse_list(fts.read())
    llb = parse_list(flb.read())

# print("load and parse complete")

tts = torch.tensor(lts, dtype=torch.float32).view(-1, 1, 28, 28).to(device)
tlb = torch.tensor(llb, dtype=torch.int32).to(device)

# print("prepare complete")

net.set_debug(True)
outputs = net(tts)
net.set_debug(False)
_, predicted = torch.max(outputs, 1)
total = tlb.size()[0]
correct = int((predicted == tlb).sum())
print(f"test_set.h and label.h test: {correct / total * 100:.4f}%")


import os
import os.path


def export_h(x, name, dname="."):
    os.makedirs(dname, exist_ok=True)
    with open(f"{os.path.join(dname, name)}.h", "w") as f:
        f.write(f"""w_t _{name}[/*{"][".join(map(str, x.size()))}*/] = {{\n""")
        f.write(",\n".join(str(float(e)) for e in x.view(-1)))
        f.write("\n};\n")


target_dir = "../data"
for u in net.chain:
    if len(u) == 2:
        f, name = u
        export_h(f.weight, f"weights_{name}", target_dir)
        export_h(f.bias, f"bias_{name}", target_dir)
