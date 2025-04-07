import torch.nn as nn
import torch.nn.functional as F

class LithologyModel(nn.Module):
    def __init__(self, args, input_size, num_classes):
        super(LithologyModel, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=args.conv_channels[0],
            kernel_size=args.kernel_size,
            stride=1,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=args.conv_channels[0],
            out_channels=args.conv_channels[1],
            kernel_size=args.kernel_size,
            stride=1,
            padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=args.conv_channels[1],
            out_channels=args.conv_channels[2],
            kernel_size=args.kernel_size,
            stride=1,
            padding=1
        )
        self.fc1 = nn.Linear(args.conv_channels[2], args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x