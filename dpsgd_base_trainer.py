import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, mean_squared_error
from dpsgd_models import LogisticRegressionModel, LinearRegressionModel


class BaseTrainer:
    def __init__(self, model, train_loader, test_loader, epochs, lr):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.lr = lr

        # Infer task type dynamically based on model class
        self.task_type = self._infer_task_type(model)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        # Set criterion and metric dynamically
        if self.task_type == "classification":
            self.criterion = nn.BCELoss()
            self.metric = self.auc
            self.model_name = "logistic_"
        else:
            self.criterion = nn.MSELoss()
            self.metric = self.mse
            self.model_name = "linear_"

        self.train_losses = []
        self.test_losses = []
        self.grad_norms = []

    def _infer_task_type(self, model):
        # Simple model check based on class name or type
        if isinstance(model, LogisticRegressionModel):
            return "classification"
        elif isinstance(model, LinearRegressionModel):
            return "regression"
        else:
            raise ValueError("Unknown model type. Please use a supported model.")

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.float(), targets.float().unsqueeze(1)
                loss = self.train_step(inputs, targets)
                running_loss += loss

            # Calculate epoch loss
            epoch_loss = running_loss / len(self.train_loader)
            self.train_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{self.epochs}, Training Loss: {epoch_loss:.4f}")

            # Evaluate on test set
            test_loss, metric_value = self.evaluate()
            self.test_losses.append(test_loss)

            metric_label = "Accuracy" if self.task_type == "classification" else "MSE"
            print(f"Epoch {epoch + 1}/{self.epochs}, Test Loss: {test_loss:.4f}, {metric_label}: {metric_value:.4f}")

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
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.float(), targets.float().unsqueeze(1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                all_targets.extend(targets.numpy().flatten())
                all_outputs.extend(outputs.numpy().flatten())

        # Calculate loss and selected metric
        test_loss = running_loss / len(self.test_loader)
        metric_value = self.metric()

        return test_loss, metric_value

    # Accuracy calculation for classification tasks
    def _accuracy(self, targets, outputs):
        predicted = [1 if o >= 0.5 else 0 for o in outputs]  # Sigmoid threshold for classification
        correct = sum([1 for p, t in zip(predicted, targets) if p == t])
        return correct / len(targets)

    # MSE calculation for regression tasks
    def mse(self):
        if self.task_type != "regression":
            raise ValueError("MSE can only be calculated for regression tasks.")

        self.model.eval()
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.float(), targets.float().unsqueeze(1)
                outputs = self.model(inputs)
                all_targets.extend(targets.numpy().flatten())
                all_outputs.extend(outputs.numpy().flatten())

        mse_value = mean_squared_error(all_targets, all_outputs)
        return mse_value

    def auc(self):
        if self.task_type != "classification":
            raise ValueError("AUC can only be calculated for classification tasks.")

        self.model.eval()
        all_targets = []
        all_outputs = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.float(), targets.float().unsqueeze(1)
                outputs = self.model(inputs)
                all_targets.extend(targets.numpy().flatten())
                all_outputs.extend(outputs.numpy().flatten())

        auc_value = roc_auc_score(all_targets, all_outputs)
        return auc_value

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.test_losses, label="Test Loss", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{self.model_name.capitalize()} Training and Test Loss Over Epochs")
        plt.legend()
        plt.show()

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
