from abc import abstractmethod


class IModel():
    # Hyperparameters
    def __init__(self) -> None:
        self.max_curvature = 1/8
        self.max_std = 0.1

    @abstractmethod
    def create_driving_model():
        pass

    @abstractmethod
    def run_driving_model():
        pass
