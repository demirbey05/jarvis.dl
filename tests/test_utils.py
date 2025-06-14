import types
from utils import add_to_class,HyperParameters

class Dummy:
    pass

@add_to_class(Dummy)
def say_hi(self):
    return "hi"

def test_add_to_class():
    assert hasattr(Dummy, "say_hi")
    assert isinstance(Dummy.say_hi, types.FunctionType)
    assert Dummy().say_hi() == "hi"


def test_hyperparams():
    class X(HyperParameters):
        def __init__(self,x,y,z) -> None:
            self.save_hyperparameters()
    
    x = X(1,2,3)
    assert x.x == 1
    assert x.y == 2
    assert x.z == 3

def test_hyperparams_ignore():
    class X(HyperParameters):
        def __init__(self,x,y,z) -> None:
            self.save_hyperparameters(ignore=['y'])
    
    x = X(1,2,3)
    assert x.x == 1
    assert hasattr(x,'x')
    assert not hasattr(x,'y')
    assert x.z == 3 
    assert hasattr(x,'z')
    
