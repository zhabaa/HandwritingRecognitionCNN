import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class DigitRecognizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
    
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
    
        return F.log_softmax(self.fc2(x), dim=1)

class DigitPredictor:
    def __init__(self, model_path=None):
        self.model = DigitRecognizer()

        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=False))

        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    
    def predict(self, image_array):
        """Принимает numpy array 28x28"""

        tensor_image = self.transform(image_array).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor_image)
            probs = torch.exp(output).numpy()[0]

        return probs, int(np.argmax(probs))
