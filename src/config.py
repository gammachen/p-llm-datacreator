import json
import os
from typing import List, Dict, Any

class Config:
    """配置类，负责加载和管理配置"""
    
    def __init__(self, config_path: str = None):
        # 默认配置
        self.api_key = "YOUR_API_KEY"
        self.base_url = "https://api.openai.com/v1"
        self.model = "gpt-3.5-turbo"
        self.temperature = 0.7
        self.max_tokens = 256
        self.retry_count = 3
        self.retry_delay = 5
        self.num_seed_examples = 3
        self.min_instruction_length = 5
        self.min_output_length = 5
        self.blacklist_keywords = ["色情", "暴力", "仇恨言论", "歧视", "政治", "宗教"]
        
        # 如果提供了配置文件路径，则加载配置
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新配置
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            print(f"成功加载配置: {config_path}")
        except Exception as e:
            print(f"加载配置失败: {e}，使用默认配置")
    
    def save_config(self, config_path: str) -> None:
        """保存配置到文件
        
        Args:
            config_path: 配置文件保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # 获取所有非内置属性
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_dict[key] = value
        
        # 保存到文件
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        print(f"配置已保存到: {config_path}")
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        config_str = "配置信息:\n"
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"  {key}: {value}\n"
        return config_str