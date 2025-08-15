import time
from typing import Dict, Any, Optional

import openai
from openai import OpenAI

class LLMClient:
    """LLM客户端，负责与语言模型API交互"""

    def __init__(self, config):
        self.config = config
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.retry_count = config.retry_count
        self.retry_delay = config.retry_delay

        # 初始化API客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本
        
        Args:
            prompt: 提示词
            **kwargs: 其他参数，会覆盖默认配置
            
        Returns:
            生成的文本
        """
        params = {
            "model": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # 重试机制
        for attempt in range(self.retry_count):
            try:
                return self._call_api(prompt, params)
            except Exception as e:
                if attempt < self.retry_count - 1:
                    print(f"API调用失败: {e}，{self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    raise e

    def _call_api(self, prompt: str, params: Dict[str, Any]) -> str:
        """调用OpenAI API"""
        response = self.client.chat.completions.create(
            model=params["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"]
        )
        return response.choices[0].message.content.strip()

    def batch_generate(self, prompts: list, **kwargs) -> list:
        """批量生成文本
        
        Args:
            prompts: 提示词列表
            **kwargs: 其他参数
            
        Returns:
            生成的文本列表
        """
        results = []
        for prompt in prompts:
            results.append(self.generate(prompt, **kwargs))
        return results