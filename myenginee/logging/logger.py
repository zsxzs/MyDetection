import logging

from typing import Optional, Union
from logging import _Level, Logger, LogRecord

from myenginee.utils import ManagerMixin


class MMLogger(Logger, ManagerMixin):
    
    def __init__(self,
                 name: str,
                 logger_name: str = 'myengine',
                 log_file: Optional[str] = None,
                 log_level: Union[int, str] = 'INFO',
                 file_mode: str = 'w',
                 distributed=False) -> None:
        Logger.__init__(self, logger_name)
        ManagerMixin.__init__(self, name)
        
        if isinstance(log_level, str):
            log_level = logging._nameToLevel[log_level]
        global_rank = _get_rank()


def print_log(msg: str,
              logger: Optional[Union[Logger, str]] = None, 
              level=logging.INFO):
    """
    根据logger进行相应的输出
        Logger类 ： logger.log
        "silent": No message will be printed.
        "current": Use latest created logger to log message.
        
    """
    
    if logger is None:
        print(msg)
    elif isinstance(logger, Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif logger == 'current':
        logger_instance = MMLogger()


def _get_rank():
    
    try:
        from myenginee.dist import get_rank
        