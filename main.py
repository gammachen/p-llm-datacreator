import argparse
import json
import os
from tqdm import tqdm

from src.generator import InstructionGenerator, InstanceGenerator
from src.filter import DataFilter
from src.utils import setup_logger, load_json, save_json
from src.config import Config

# 设置日志
logger = setup_logger()

def parse_args():
    parser = argparse.ArgumentParser(description='Self-Instruct数据生成框架')
    parser.add_argument('--config', type=str, default='config/default.json', help='配置文件路径')
    parser.add_argument('--seed_file', type=str, default='data/seed_instructions.json', help='种子指令文件')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--iterations', type=int, default=5, help='迭代次数')
    parser.add_argument('--num_per_iter', type=int, default=100, help='每轮生成指令数量')
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = Config(args.config)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载种子指令
    logger.info(f"加载种子指令: {args.seed_file}")
    seed_data = load_json(args.seed_file)
    
    # 初始化生成器和过滤器
    instruction_generator = InstructionGenerator(config)
    instance_generator = InstanceGenerator(config)
    data_filter = DataFilter(config)
    
    # 初始化数据池
    current_pool = seed_data.copy()
    logger.info(f"初始种子指令数量: {len(current_pool)}")
    
    # 迭代生成
    for iter_idx in range(args.iterations):
        logger.info(f"开始第 {iter_idx+1}/{args.iterations} 轮迭代")
        
        # 1. 生成新指令
        logger.info(f"生成新指令...")
        new_instructions = instruction_generator.generate(
            current_pool, 
            num_to_generate=args.num_per_iter
        )
        logger.info(f"生成了 {len(new_instructions)} 条新指令")
        
        # 2. 为指令生成输入-输出对
        logger.info(f"为指令生成输入-输出对...")
        new_data = []
        for inst in tqdm(new_instructions):
            instance = instance_generator.generate(inst)
            if instance and data_filter.is_valid(instance):
                new_data.append(instance)
        
        logger.info(f"成功生成 {len(new_data)}/{len(new_instructions)} 条有效数据")
        
        # 3. 去重并加入数据池
        old_pool_size = len(current_pool)
        current_pool.extend(new_data)
        
        # 4. 保存当前迭代结果
        iter_output_file = os.path.join(args.output_dir, f"self_instruct_iter_{iter_idx}.json")
        save_json(current_pool, iter_output_file)
        logger.info(f"保存迭代结果到: {iter_output_file}")
        logger.info(f"当前数据池大小: {len(current_pool)} (新增 {len(current_pool) - old_pool_size} 条)")
    
    # 保存最终结果
    final_output_file = os.path.join(args.output_dir, "self_instruct_final.json")
    save_json(current_pool, final_output_file)
    logger.info(f"生成完成！最终数据集大小: {len(current_pool)}")
    logger.info(f"最终数据保存至: {final_output_file}")

if __name__ == "__main__":
    main()