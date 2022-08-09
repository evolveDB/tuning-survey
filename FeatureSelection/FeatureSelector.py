import abc
class FeatureSelector(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def fit(self,data):
        pass

    @abc.abstractmethod
    def transform(self,Xnew):
        pass
        