import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend(['.', '..', path])
print(sys.path)

from pathlib import Path
from typing import Optional, Union, Tuple
from addict import Dict
import tempfile
import platform
import shutil

from myenginee.utils import check_file_exist
# from utils import check_file_exist

RESERVED_KEYS = ['filename', 'text', 'pretty_text', 'env_variables']

if platform.system() == 'Windows':
    import regex as re
else:
    import re  # type: ignore

class ConfigDict(Dict):
    """
    配置字典：其接口与python的内置字典相同，可作为普通字典使用。
    """
    def __missing__(self, name):
        return KeyError(name)
    
    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        except Exception as e:
            raise e
        else:
            return value

class Config:
    def __init__(self,
                 cfg_dict: dict = None,
                 cfg_text: Optional[str] = None,
                 filename: Optional[Union[str, Path]] = None,
                 env_variables: Optional[dict] = None):
        filename = str(filename) if isinstance(filename, Path) else filename
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            # 判断是否是dict
            raise TypeError(f'cfg_dict must be a dict, but got {type(cfg_dict)}')
        # 判断关键字名是否合法
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')
            
        super().__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super().__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, encoding='utf-8') as f:
                text = f.read()
        else:
            text = ''
        super().__setattr__('_text', text)
        if env_variables is None:
            env_variables = dict()
        super().__setattr__('_env_variables', env_variables)
    
    @staticmethod
    def fromfile(filename: Union[str, Path],
                 use_predefined_variables: bool = True,
                 import_custom_modules: bool = True,
                 use_environment_variables: bool = True) -> 'Config':
        """
        通过配置文件，构建一个Config实例
        """
        filename = str(filename) if isinstance(filename, Path) else filename
        cfg_dict, cfg_text, env_variables = Config._file2dict(
            filename, use_predefined_variables, use_environment_variables)
    
    @staticmethod
    def _file2dict(filename: str,
                   use_predefined_variables: bool = True,
                   use_environment_variables: bool = True) -> Tuple[dict, str, dict]:
        """
        将文件内容转化成字典
        """
        # 得到配置文件的绝对路径，并检查文件是否存在
        filename = os.path.abspath(os.path.expanduser(filename))
        check_file_exist(filename)
        # 支持py、json和yaml文件
        fileExtname = os.path.splitext(filename)[1] 
        if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
            raise OSError('Only py/yml/yaml/json type are supported now!')
        
        with tempfile.TemporaryDirectory() as temp_config_dir: # 生成临时目录
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname)       # 生成与filename后缀相同的临时文件
            if platform.system() == 'Windows': 
                temp_config_file.close()  # ?
            
            # Substitute predefined variables
            # 设置参数文件名
            if use_predefined_variables:
                 Config._substitute_predefined_vars(filename, temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)
                
            # Substitute environment variables
            env_variables = dict()
            if use_environment_variables:
                env_variables = Config._substitute_env_variables(
                    temp_config_file.name, temp_config_file.name)
            # TODO
                
            
                
    @staticmethod
    def _substitute_predefined_vars(filename: str, temp_config_name: str):
        """
        设置配置文件中的某些变量（设置与配置文件名称相关的目录或文件）
        
        比如说，我们想将配置文件中的某些值和当前配置文件的名字或者路径有关。
        （场景1：训练模型保存在相同名称的目录下）
        首先，我们会在配置文件中预先设置：
        work_dir = '. /work_dir/{{ fileBasenameNoExtension }}'
        然后利用这个函数替换，比如配置文件名为config_setting1.py，
        经过替换， work_dir = '. /work_dir/config_setting1'
        """
        file_dirname = os.path.join(filename)
        file_basename = os.path.basename(filename)
        file_basename_splitext = os.path.splitext(file_basename)
        file_basename_no_extension = file_basename_splitext[0]
        file_extname = file_basename_splitext[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)
        with open(filename, encoding='utf-8') as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)
            
        @staticmethod
        def _substitute_env_variables(filename: str, temp_config_name: str):
            """
            替换一些默认的环境变量
            """
            with open(filename, encoding='utf-8') as f:
                config_file = f.read()
            regexp = r'\{\{[\'\"]?\s*\$(\w+)\s*\:\s*(\S*?)\s*[\'\"]?\}\}'
            keys = re.findall(regexp, config_file)
            env_variables = dict()
            for var_name, value in keys:
                regexp = r'\{\{[\'\"]?\s*\$' + var_name + r'\s*\:\s*' + value + r'\s*[\'\"]?\}\}'
                if var_name in os.environ:
                    value = os.environ[var_name]
                    env_variables[var_name] = value
                    print_log
            
        
        
        
        
if __name__ == '__main__':
    

    
    c = Config.fromfile('/home/jykj/zs/mydetection/faster-rcnn_r50_fpn_1x_coco.py')
    
    