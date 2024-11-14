import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data Preparation (same for both)
def prepare_data_loaders(X, y, X_test, y_test, batch_size=32, scale_data=False, shuffle=False):
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_test = scaler.transform(X_test)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    train_dataset = TensorDataset(X_tensor, y_tensor)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


# base logistic regression model, valid for both sgd and dpsgd
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# base trainer
class LogisticRegressionBaseTrainer:
    def __init__(self, model, train_loader, test_loader, epochs, lr):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.criterion = nn.BCELoss()
        self.train_losses = []
        self.test_losses = []
        self.grad_norms = []
        self.model_name = "sgd_"

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.float(), targets.float().unsqueeze(1)
                loss = self.train_step(inputs, targets)
                running_loss += loss

            # train loss
            epoch_loss = running_loss / len(self.train_loader)
            self.train_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{self.epochs}, Training Loss: {epoch_loss:.4f}")

            # test loss
            test_loss, accuracy = self.evaluate()  # Modified to return test loss
            self.test_losses.append(test_loss)  # Store the test loss
            print(f"Epoch {epoch + 1}/{self.epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")

    def train_step(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.float(), targets.float().unsqueeze(1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()
                predicted = outputs.round()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        test_loss = running_loss / len(self.test_loader)
        accuracy = correct / total
        return test_loss, accuracy  # Return test loss and accuracy

    def auc(self):
        self.model.eval()
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.float(), targets.float().unsqueeze(1)
                outputs = self.model(inputs)
                all_targets.extend(targets.numpy().flatten())
                all_outputs.extend(outputs.numpy().flatten())

        auc = roc_auc_score(all_targets, all_outputs)
        print(f"Test AUC: {auc:.4f}")
        return auc

    def calculate_grad_norm(self):
        # Calculate the L2 norm of gradients for each parameter and return the average
        total_norm = 0.0
        count = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item()
                count += 1
        return total_norm / count if count > 0 else 0.0

    def plot_loss(self):
        filename = self.model_name + "losses.png"
        plt.figure()
        plt.plot(range(len(self.train_losses)), self.train_losses, label="Training Loss")
        plt.plot(range(len(self.test_losses)), self.test_losses, label="Test Loss", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Test Loss Over Epochs")
        plt.legend()
        plt.savefig(filename)
        plt.close()  # Close the plot to free up memory

    def plot_grad_norms(self):
        plt.plot(range(len(self.grad_norms)), self.grad_norms, label="Average Gradient Norm")
        plt.xlabel("Epochs")
        plt.ylabel("Average L2 Norm of Gradients")
        plt.title("Average Gradient Norm Over Epochs")
        plt.legend()
        plt.show()

    def plot_grad_norms(self, filename="grad_norms_plot.png"):
        plt.figure()
        plt.plot(range(len(self.grad_norms)), self.grad_norms, label="Average Gradient Norm")
        plt.xlabel("Epochs")
        plt.ylabel("Average L2 Norm of Gradients")
        plt.title("Average Gradient Norm Over Epochs")
        plt.legend()
        plt.savefig(filename)
        plt.close()  # Close the plot to free up memory

    def compare_loss_with_custom(self, X, y):
        """
        Compares the PyTorch loss with the custom NumPy-based loss for the whole dataset.

        Parameters:
            X (torch.Tensor): The input features for the dataset.
            y (torch.Tensor): The target labels for the dataset.
        """
        # forward pass with PyTorch
        self.model.eval()
        with torch.no_grad():
            inputs, targets = X.float(), y.float().unsqueeze(1)
            outputs = self.model(inputs)
            pytorch_loss = self.criterion(outputs, targets).item()
            print(f"PyTorch computed loss: {pytorch_loss}")

        theta = np.concatenate([p.detach().numpy().flatten() for p in self.model.parameters()])
        x_np = X.numpy()
        y_np = y.numpy().flatten()

        # custom loss function
        def custom_loss(theta, x, y):
            m = x.shape[0]
            h = LogisticRegressionObjective.hypothesis(theta, x)
            return -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

        # Calculate custom loss
        custom_loss_value = custom_loss(theta, x_np, y_np)
        print(f"Custom computed loss: {custom_loss_value}")

        # Compare both losses
        difference = abs(pytorch_loss - custom_loss_value)
        print(f"Difference between PyTorch and custom loss: {difference}")


# Non-Private Trainer (inherits from Base Trainer)
class NonPrivateSGDTrainer(LogisticRegressionBaseTrainer):
    def __init__(self, model, train_loader, test_loader, epochs, lr):
        super().__init__(model, train_loader, test_loader, epochs, lr)


# DP Trainer (inherits from Base Trainer, with DP-specific train_step)
class DPSGDTrainer(LogisticRegressionBaseTrainer):
    def __init__(self, model, train_loader, test_loader, epochs, lr, epsilon, delta, max_grad_norm, use_dp):
        super().__init__(model, train_loader, test_loader, epochs, lr)
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.sample_size = len(train_loader.dataset)
        self.batch_size = train_loader.batch_size
        self.epochs = epochs
        # Uncomment below if using differential privacy
        self.use_dp = use_dp
        if self.use_dp:
            self.model_name = "dp-sgd_"
            self._setup_dp()


    # Uncomment if using Opacus for DP
    def _setup_dp(self):
        privacy_engine = PrivacyEngine()
        self.noise_multiplier = get_noise_multiplier(target_epsilon=self.epsilon, target_delta=self.delta,
                                                sample_rate=self.batch_size / self.sample_size,
                                                epochs=self.epochs)

        print(self.optimizer)
        self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm
        )
        print(self.optimizer)

        from opacus.optimizers import DPOptimizer

        # Assuming `optimizer` is your optimizer variable
        if isinstance(self.optimizer, DPOptimizer):
            print("The optimizer is wrapped by Opacus and is a DPOptimizer.")
        else:
            print("The optimizer is NOT wrapped by Opacus.")

    def train_step(self, inputs, targets):
        # Differentially Private SGD step
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        loss.backward()
        self.optimizer.step()

        return loss.item()
