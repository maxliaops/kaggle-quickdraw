import torch.nn as nn

from models.alexnet import alexnet


class AlexNetWrapper(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.alexnet = alexnet(pretrained=True)

        classifier = list(self.alexnet.classifier.children())[:-1]
        print(list(self.alexnet.classifier.children())[-1:])
        classifier.append(nn.Linear(4096, num_classes))
        self.alexnet.classifier = nn.Sequential(*classifier)
        print(self.alexnet.classifier)

    def forward(self, x):
        return self.alexnet(x)
