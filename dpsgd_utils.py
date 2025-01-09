import torch
from torch.utils.data import TensorDataset, DataLoader

def prepare_data_loaders(X, y, X_test, y_test, batch_size, scale_data=False, shuffle=False):
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


def advanced_composition(epsilon_0, delta_0, k):
    """
    Converts budget used for single hyperparameter search runs to overall budget spent under
    advanced composition (T. Steinke, 2022, Theorem 22)

    k: size of the search space
    epsilon_0, delta_0: privacy parameters effectively used for each run
    """

    epsilon = min( k*epsilon_0, 1/2 * k * epsilon_0**2 + np.sqrt(2 * np.log(1/(k * delta_0) * k*epsilon_0**2)) )
    delta = 2 * k * delta_0

    return epsilon


def inverse_advanced_composition(epsilon, delta, k):
    """
    Returns epsilon_0 and delta_0 from overall budget epsilon and delta
    under advanced composition.

    k: size of the search space
    epsilon, delta: overall privacy parameters
    """
    # Compute delta_0 from delta
    delta_0 = delta / (2 * k)

    # Solve for epsilon_0
    # Case 1: epsilon = k * epsilon_0
    epsilon_0_case1 = epsilon / k

    # Case 2: epsilon = 1/2 * k * epsilon_0^2 + sqrt(2 * log(1 / (k * delta_0))) * k * epsilon_0^2
    # Solve the quadratic equation ax^2 + bx + c = 0
    log_term = np.sqrt(2 * np.log(1 / (k * delta_0)))
    a = 0.5 * k
    b = k * log_term
    c = -epsilon

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No solution for epsilon_0 in the given parameters.")

    # only positive root is valid
    epsilon_0_case2 = (-b + np.sqrt(discriminant)) / (2 * a)

    epsilon_0 = min(epsilon_0_case1, epsilon_0_case2)

    return epsilon_0, delta_0