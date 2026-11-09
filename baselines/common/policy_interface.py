from abc import ABC,abstractmethod

class BasePolicy(ABC):
 """Receives RGB and verbatim instruction only; returns native model action.

 Simulator state may be logged by the evaluator, never passed through this
 boundary. Conversion to a canonical simulator action belongs in ActionAdapter.
 """
 @abstractmethod
 def reset(self):pass
 @abstractmethod
 def predict_action(self,observation):pass
 @abstractmethod
 def load_checkpoint(self,path):pass
