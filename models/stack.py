import torch
import torch.nn as nn

from models import SeNet
from utils import freeze


class StackNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model0 = StackNet.load_model(0)
        self.model1 = StackNet.load_model(1)
        self.model2 = StackNet.load_model(2)
        self.model3 = StackNet.load_model(3)
        self.model4 = StackNet.load_model(4)

        self.fc0 = nn.Linear(num_classes, 512)
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        out0 = self.model0(x)
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        out4 = self.model4(x)

        out = torch.cat([out0, out1, out2, out3, out4], dim=1)

        return self.fc1(self.fc0(out))

    @staticmethod
    def load_model(category_shard):
        model = nn.DataParallel(SeNet("seresnext50", 68))
        model.load_state_dict(
            torch.load("/storage/models/quickdraw/cat_{}/model.pth".format(category_shard),
                       map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        freeze(model)
        return model
