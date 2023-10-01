import math
import torch.nn as nn
from torchvision import models
import torch

class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

def get_model(model_type=1):
    if model_type == 1:
        model = Model()
    elif model_type == 2:
        model = models.resnet18(weights=None, num_classes=10)
    elif model_type == 3: # Improved ResNet18
        model = models.resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)  # set kernel of the first CNN as 3*3
        model.maxpool = nn.MaxPool2d(1, 1, 0)  # maxpooling layer ignores too much information; use 1*1 maxpool to diable pooling layer

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model
    
def model_norm(model_1, model_2):
	squared_sum = 0
	for name, layer in model_1.named_parameters():
		squared_sum += torch.sum(torch.pow(layer.data - model_2.state_dict()[name].data, 2))
	return math.sqrt(squared_sum)
    
if __name__ == '__main__':
    model = get_model(model_type=1)

    for name, weights in model.state_dict().items():
        print(name, weights.shape)