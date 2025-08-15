#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级指令数据生成示例

这个脚本展示了Self-Instruct框架的高级功能，包括：
1. 语义去重
2. 多样性控制
3. 多轮迭代
4. 质量评估
"""

import sys
import os
import json
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generator import InstructionGenerator, InstanceGenerator
from src.filter import DataFilter
from src.config import Config
from src.utils import save_json, load_json, semantic_deduplicate

def main():
    # 加载配置
    config = Config('../config/default.json')
    
    # 设置API密钥（也可以在配置文件中设置）
    config.api_key = os.environ.get('OPENAI_API_KEY', config.api_key)
    
    # 自定义配置
    config.temperature = 0.8  # 增加多样性
    config.num_seed_examples = 5  # 每次使用5个种子示例
    
    # 加载种子指令
    with open('../data/seed_instructions.json', 'r', encoding='utf-8') as f:
        seed_data = json.load(f)
    
    print(f"加载了 {len(seed_data)} 条种子指令")
    
    # 初始化生成器和过滤器
    instruction_generator = InstructionGenerator(config)
    instance_generator = InstanceGenerator(config)
    data_filter = DataFilter(config)
    
    # 创建输出目录
    output_dir = 'output/advanced'
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化数据池
    current_pool = seed_data.copy()
    
    # 多轮迭代
    iterations = 3
    num_per_iter = 10
    
    for iter_idx in range(iterations):
        print(f"\n开始第 {iter_idx+1}/{iterations} 轮迭代")
        
        # 1. 生成新指令
        print(f"生成新指令...")
        new_instructions = instruction_generator.generate(
            current_pool, 
            num_to_generate=num_per_iter
        )
        print(f"生成了 {len(new_instructions)} 条新指令")
        
        # 2. 为指令生成输入-输出对
        print(f"为指令生成输入-输出对...")
        new_data = []
        for inst in tqdm(new_instructions):
            instance = instance_generator.generate(inst)
            if instance and data_filter.is_valid(instance):
                new_data.append(instance)
        
        print(f"成功生成 {len(new_data)}/{len(new_instructions)} 条有效数据")
        
        # 3. 语义去重
        try:
            unique_data = semantic_deduplicate(new_data, current_pool, threshold=0.8)
            print(f"去重后剩余 {len(unique_data)}/{len(new_data)} 条数据")
        except ImportError:
            print("警告: 语义去重需要安装sentence-transformers库")
            unique_data = new_data
        
        # 4. 加入数据池
        old_pool_size = len(current_pool)
        current_pool.extend(unique_data)
        
        # 5. 保存当前迭代结果
        iter_output_file = os.path.join(output_dir, f"iter_{iter_idx}.json")
        save_json(current_pool, iter_output_file)
        print(f"保存迭代结果到: {iter_output_file}")
        print(f"当前数据池大小: {len(current_pool)} (新增 {len(current_pool) - old_pool_size} 条)")
    
    # 保存最终结果
    final_output_file = os.path.join(output_dir, "final_dataset.json")
    save_json(current_pool, final_output_file)
    print(f"\n生成完成！最终数据集大小: {len(current_pool)}")
    print(f"最终数据保存至: {final_output_file}")
    
    # 数据统计
    print("\n数据统计:")
    print(f"  总指令数: {len(current_pool)}")
    print(f"  平均指令长度: {sum(len(d['instruction']) for d in current_pool) / len(current_pool):.1f} 字符")
    print(f"  平均输出长度: {sum(len(d['output']) for d in current_pool) / len(current_pool):.1f} 字符")

if __name__ == "__main__":
    main()