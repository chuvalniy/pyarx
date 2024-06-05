import torch


class KalmanFilter:
    def __init__(self, velocity):
        self.velocity = velocity

        self.sigma_psi = 1.0  # TODO: Define as difference between ideal state and sensor state
        self.sigma_eta = 1.0  # TODO: Define as difference between ideal state and sensor state

        self.x = [0.0]
        self.z = [0.0 + torch.randn(mean=0.0, std=self.sigma_eta).item()]

    def predict(self):
        new_x = self.x[-1] + self.velocity + torch.randn(mean=0.0, std=self.sigma_psi).item()
        new_z = new_x + torch.randn(mean=0.0, std=self.sigma_eta).item()

        self.x.append(new_x)
        self.z.append(new_z)

    def update(self):
        pass
