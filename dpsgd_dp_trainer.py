import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant  # For RDP accounting
from opacus.accountants.utils import get_noise_multiplier
from dpsgd_base_trainer import BaseTrainer


# DP Trainer (inherits from Base Trainer, with DP-specific train_step)
class DPSGDTrainer(BaseTrainer):
    def __init__(self, model, train_loader, test_loader, epochs, lr, epsilon, delta, max_grad_norm, use_dp, rdp_alphas):
        super().__init__(model, train_loader, test_loader, epochs, lr)
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.sample_size = len(train_loader.dataset)
        self.batch_size = train_loader.batch_size
        self.epochs = epochs
        # Uncomment below if using differential privacy
        self.use_dp = use_dp
        self.rdp_alphas = rdp_alphas
        self.all_epsilons = []
        self.smallest_alpha = None

        if self.use_dp:
            self.model_name = "dp-sgd_"
            self._setup_dp()

    # Uncomment if using Opacus for DP
    def _setup_dp(self):
        privacy_engine = PrivacyEngine()

        accountant = RDPAccountant(alphas=self.rdp_alphas)

        self.noise_multiplier, self.alpha, self.all_epsilons = get_noise_multiplier(target_epsilon=self.epsilon, target_delta=self.delta,
                                                sample_rate=self.batch_size / self.sample_size,
                                                epochs=self.epochs,
                                                steps=None,
                                                accountant=accountant
                                                     )


        valid_alphas = [(alpha, epsilon) for alpha, epsilon in self.all_epsilons if epsilon <= self.epsilon]
        self.smallest_alpha = min(valid_alphas, key=lambda x: x[0])

        self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm
        )
        print(self.optimizer)


    def train_step(self, inputs, targets):
        # Differentially Private SGD step
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        loss.backward()
        self.optimizer.step()

        return loss.item()