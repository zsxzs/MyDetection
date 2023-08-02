import inspect
import threading
import warnings

from collections import OrderedDict
from typing import Type, TypeVar

_lock = threading.RLock()
T = TypeVar('T')

def _accquire_lock() -> None:
    """加锁"""
    if _lock:
        _lock.acquire()
        
def _release_lock() -> None:
    """释放锁"""
    if _lock:
        _lock.release()

class ManagerMate(type):
    """
    所有由ManagerMate生成的类都要有name这个参数
    """
    def __init__(cls, *args): # __init__(self, *args)
        cls.__instance_dict = OrderedDict()
        params = inspect.getfullargspec(cls) # 获取类__init__参数的名称
        params_names = params[0] if params[0] else []
        assert 'name' in params_names, f'{cls} must have the `name` argument'
        super().__init__(*args) # type.__init__(class, classname, superclasses, attributedict)
        
class ManagerMixin(metaclass=ManagerMate):
    
    def __init__(self, 
                 name: str,
                 **kwargs):
        assert isinstance(name, str) and name, \
            'name argument must be an non-empty string.'
        self._instance_name = name
        
    @classmethod
    def get_instance(cls: Type[T], 
                     name: str,
                     **kwargs) -> T:
        _accquire_lock()
        assert isinstance(name, str), \
            f'type of name should be str, but got {type(cls)}'
        instance_dict = cls._instance_dict
        
        if name not in instance_dict:
            instance = cls(name=name, **kwargs)
            instance_dict[name] = instance
        elif kwargs:
            warnings.warn(
                f'{cls} instance named of {name} has been created, '
                'the method `get_instance` should not accept any other '
                'arguments')
        _release_lock()
        return instance_dict[name]
    
    @classmethod
    def get_current_instance(cls):
        _accquire_lock()
        if not cls._instance_dict:
            raise RuntimeError(
                f'Before calling {cls.__name__}.get_current_instance(), you '
                'should call get_instance(name=xxx) at least once.')
        name = next(iter(reversed(cls._instance_dict)))
        _release_lock()
        return cls._instance_dict[name]
    
    @classmethod
    def check_instance_created(cls, name: str) -> bool:
        return name in cls.__instance_dict
    
    @property
    def instance_name(self) -> str:
            return self._instance_name
        