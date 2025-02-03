from abc import abstractmethod, ABC


class BaseCostModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass
