from typing import Dict, List, Any
import re

class DataFilter:
    """数据过滤器，负责清洗低质量数据"""
    
    def __init__(self, config):
        self.config = config
        self.blacklist = config.blacklist_keywords
        self.min_instruction_length = config.min_instruction_length
        self.min_output_length = config.min_output_length
        self.invalid_outputs = ["n/a", "我不知道", "不知道", "无法回答", "无法提供", "抱歉", ""]
    
    def is_valid(self, instance: Dict[str, str]) -> bool:
        """检查数据实例是否有效
        
        Args:
            instance: 包含指令、输入和输出的字典
            
        Returns:
            数据是否有效
        """
        # 规则1：指令长度检测
        if len(instance["instruction"]) < self.min_instruction_length:
            return False
        
        # 规则2：输出长度检测
        if len(instance["output"]) < self.min_output_length:
            return False
        
        # 规则3：输出相关性检测
        if instance["output"].lower() in self.invalid_outputs:
            return False
        
        # 规则4：关键词黑名单过滤
        if self._contains_blacklist_keywords(instance["instruction"]):
            return False
        
        # 规则5：输出中不应包含抱歉、歉意等表达
        if self._contains_apology(instance["output"]):
            return False
        
        return True
    
    def _contains_blacklist_keywords(self, text: str) -> bool:
        """检查文本是否包含黑名单关键词"""
        return any(word in text.lower() for word in self.blacklist)
    
    def _contains_apology(self, text: str) -> bool:
        """检查输出是否包含道歉或拒绝回答的表达"""
        apology_patterns = [
            r"抱歉", r"对不起", r"很遗憾", r"无法回答", r"无法提供",
            r"sorry", r"apologize", r"cannot answer", r"can't provide"
        ]
        return any(re.search(pattern, text.lower()) for pattern in apology_patterns)
    
    def filter_batch(self, instances: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """批量过滤数据
        
        Args:
            instances: 数据实例列表
            
        Returns:
            过滤后的数据实例列表
        """
        return [inst for inst in instances if self.is_valid(inst)]