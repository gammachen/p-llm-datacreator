import json
import random
from typing import List, Dict, Any

from .llm import LLMClient
from .utils import deduplicate_instructions

class InstructionGenerator:
    """指令生成器，负责基于种子指令生成新的指令"""
    
    def __init__(self, config):
        self.config = config
        self.llm_client = LLMClient(config)
        self.prompt_template = """
你是一个指令生成器。请基于以下示例生成{num_prompts}条新的、多样化的任务指令：
{seed_examples}

新指令要求：
1. 避免重复示例中的指令
2. 覆盖不同领域（写作、翻译、编码等）
3. 使用自然语言描述任务
4. 每条指令应该是独立的，不要有编号
5. 特别注意：生成至少三种不同任务类型（如创意写作、信息提取、逻辑推理）

生成的指令列表：
"""
    
    def generate(self, seed_data: List[Dict[str, Any]], num_to_generate: int = 10) -> List[str]:
        """生成新指令
        
        Args:
            seed_data: 种子数据列表
            num_to_generate: 要生成的指令数量
            
        Returns:
            生成的新指令列表
        """
        # 随机选择种子示例
        sample_size = min(self.config.num_seed_examples, len(seed_data))
        seed_samples = random.sample(seed_data, sample_size)
        
        # 提取指令部分
        seed_examples = "\n".join([f"- {d['instruction']}" for d in seed_samples])
        
        # 构建提示词
        prompt = self.prompt_template.format(
            num_prompts=num_to_generate,
            seed_examples=seed_examples
        )
        
        # 调用LLM生成
        response = self.llm_client.generate(prompt)
        
        # 解析响应获取指令列表
        instructions = self._parse_instructions(response)
        
        # 去重
        existing_instructions = [d["instruction"] for d in seed_data]
        unique_instructions = deduplicate_instructions(instructions, existing_instructions)
        
        return unique_instructions
    
    def _parse_instructions(self, response: str) -> List[str]:
        """解析LLM响应，提取指令列表"""
        instructions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            # 移除行首的序号和符号
            if line.startswith(("-", "*", "•")):
                line = line[1:].strip()
            elif any(line.startswith(f"{i}.") for i in range(1, 100)):
                line = line.split(".", 1)[1].strip()
            
            if line and len(line) > 5:  # 简单过滤太短的指令
                instructions.append(line)
        
        return instructions


class InstanceGenerator:
    """实例生成器，负责为指令生成输入-输出对"""
    
    def __init__(self, config):
        self.config = config
        self.llm_client = LLMClient(config)
        self.prompt_template = """
根据指令生成输入和输出：
指令：{instruction}

要求：
1. 如果任务不需要输入，填写"无"
2. 输出必须直接完成任务
3. 输入应该是真实、多样化的
4. 输出应该是高质量、有帮助的

请按以下格式回复：
输入：<在此生成任务输入>
输出：<在此生成任务输出>
"""
    
    def generate(self, instruction: str) -> Dict[str, str]:
        """为指令生成输入-输出对
        
        Args:
            instruction: 指令文本
            
        Returns:
            包含指令、输入和输出的字典，如果生成失败则返回None
        """
        # 构建提示词
        prompt = self.prompt_template.format(instruction=instruction)
        
        # 调用LLM生成
        response = self.llm_client.generate(prompt)
        
        # 解析响应
        try:
            instance = self._parse_instance(instruction, response)
            return instance
        except Exception as e:
            print(f"解析实例失败: {e}")
            return None
    
    def _parse_instance(self, instruction: str, response: str) -> Dict[str, str]:
        """解析LLM响应，提取输入和输出"""
        input_text = ""
        output_text = ""
        
        # 解析输入和输出
        if "输入：" in response and "输出：" in response:
            parts = response.split("输出：")
            output_text = parts[1].strip()
            input_text = parts[0].replace("输入：", "").strip()
        else:
            # 尝试其他可能的格式
            if "Input:" in response and "Output:" in response:
                parts = response.split("Output:")
                output_text = parts[1].strip()
                input_text = parts[0].replace("Input:", "").strip()
        
        # 验证解析结果
        if not output_text:
            raise ValueError("无法解析输出")
        
        return {
            "instruction": instruction,
            "input": input_text if input_text else "无",
            "output": output_text
        }