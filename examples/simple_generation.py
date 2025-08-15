#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的指令数据生成示例

这个脚本展示了如何使用Self-Instruct框架生成少量指令数据
"""

import sys
import os
import json

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generator import InstructionGenerator, InstanceGenerator
from src.filter import DataFilter
from src.config import Config
from src.utils import save_json

def main():
    # 加载配置
    config = Config('../config/default.json')
    
    # 设置API密钥（也可以在配置文件中设置）
    config.api_key = os.environ.get('OPENAI_API_KEY', config.api_key)
    
    # 加载种子指令
    with open('../data/seed_instructions.json', 'r', encoding='utf-8') as f:
        seed_data = json.load(f)
    
    print(f"加载了 {len(seed_data)} 条种子指令")
    
    # 初始化生成器和过滤器
    instruction_generator = InstructionGenerator(config)
    instance_generator = InstanceGenerator(config)
    data_filter = DataFilter(config)
    
    # 生成新指令
    print("生成新指令...")
    new_instructions = instruction_generator.generate(seed_data, num_to_generate=5)
    print(f"生成了 {len(new_instructions)} 条新指令:")
    for i, inst in enumerate(new_instructions):
        print(f"  {i+1}. {inst}")
    
    # 为指令生成输入-输出对
    print("\n为指令生成输入-输出对...")
    new_data = []
    for inst in new_instructions:
        print(f"处理指令: {inst}")
        instance = instance_generator.generate(inst)
        if instance and data_filter.is_valid(instance):
            new_data.append(instance)
            print(f"  输入: {instance['input']}")
            print(f"  输出: {instance['output']}")
            print()
    
    # 保存结果
    if not os.path.exists('output'):
        os.makedirs('output')
    
    output_file = 'output/example_generated.json'
    save_json(new_data, output_file)
    print(f"\n成功生成 {len(new_data)} 条数据，保存到 {output_file}")

if __name__ == "__main__":
    main()