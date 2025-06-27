import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .model import DigitRecognizer
from config import WEIGHT_DIR


class MNISTTrainer:
    def __init__(self):
        self.BATCH_SIZE = 64
        self.EPOCHS = 3
        self.LEARNING_RATE = 0.002
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        self.train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=self.transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

        self.model = DigitRecognizer().to(self.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

        self.best_accuracy = 0

    def train(self, epoch):
        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.DEVICE), target.to(self.DEVICE)
            self.optimizer.zero_grad()
        
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            if not batch_idx % 200:
                print(
                    f"Train Epoch: \t\t {epoch} \t\t"
                    f"{batch_idx * len(data)}/{len(self.train_loader.dataset)} \t\t"  # type: ignore
                    f"Loss: {loss.item():.4f}"
                )

    def test(self):
        self.model.eval()

        test_loss = 0
        correct = 0
    
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.DEVICE), target.to(self.DEVICE)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset) # type: ignore
        accuracy = 100.0 * correct / len(self.test_loader.dataset) # type: ignore

        print(
            f"\nTest set: \t\t"
            f"Average loss: {test_loss:.4f} \t\t"
            f"Accuracy: {correct}/{len(self.test_loader.dataset)} ({accuracy:.2f}%)\n"  # type: ignore
        )
    
        return accuracy

    def run_training(self):

        for epoch in range(1, self.EPOCHS + 1):
            self.train(epoch)

            current_accuracy = self.test()

            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                torch.save(self.model.state_dict(), f"{WEIGHT_DIR}/best_model_weights.bin")

                print(f"Best model saved. Accuracy: {self.best_accuracy:.2f}%")

        torch.save(self.model.state_dict(), f"{WEIGHT_DIR}/last_model_weights.bin")
        print(
            f"Training completed!\n" \
            f"Weights saved to {WEIGHT_DIR} best and last model weights .bin files\n")


trainer = MNISTTrainer()
trainer.run_training()
