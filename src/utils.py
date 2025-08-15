import json
import logging
import os
from typing import List, Dict, Any

def setup_logger():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('self_instruct.log')
        ]
    )
    return logging.getLogger('self_instruct')

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        JSON数据
    """
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: List[Dict[str, Any]], file_path: str) -> None:
    """保存JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def deduplicate_instructions(new_instructions: List[str], existing_instructions: List[str]) -> List[str]:
    """指令去重
    
    Args:
        new_instructions: 新生成的指令列表
        existing_instructions: 已有的指令列表
        
    Returns:
        去重后的新指令列表
    """
    # 简单的文本匹配去重
    unique_instructions = []
    existing_lower = [inst.lower() for inst in existing_instructions]
    
    for inst in new_instructions:
        if inst.lower() not in existing_lower:
            unique_instructions.append(inst)
            existing_lower.append(inst.lower())
    
    return unique_instructions

def semantic_deduplicate(new_data: List[Dict[str, Any]], pool: List[Dict[str, Any]], threshold: float = 0.8):
    """语义去重
    
    Args:
        new_data: 新数据
        pool: 已有数据池
        threshold: 相似度阈值
        
    Returns:
        去重后的新数据
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # 加载模型
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 提取指令
        pool_instructions = [d["instruction"] for d in pool]
        new_instructions = [d["instruction"] for d in new_data]
        
        # 计算嵌入
        pool_embeddings = model.encode(pool_instructions)
        new_embeddings = model.encode(new_instructions)
        
        # 计算相似度并过滤
        unique_data = []
        for i, new_item in enumerate(new_data):
            # 计算与池中所有指令的相似度
            similarities = np.dot(new_embeddings[i], pool_embeddings.T) / \
                          (np.linalg.norm(new_embeddings[i]) * np.linalg.norm(pool_embeddings, axis=1))
            
            # 如果最大相似度低于阈值，则认为是唯一的
            if np.max(similarities) < threshold:
                unique_data.append(new_item)
        
        return unique_data
    except ImportError:
        print("警告: sentence-transformers未安装，使用简单文本匹配去重")
        return deduplicate_instructions_dict(new_data, pool)

def deduplicate_instructions_dict(new_data: List[Dict[str, Any]], pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """基于字典的指令去重"""
    unique_data = []
    pool_instructions = [d["instruction"].lower() for d in pool]
    
    for item in new_data:
        if item["instruction"].lower() not in pool_instructions:
            unique_data.append(item)
            pool_instructions.append(item["instruction"].lower())
    
    return unique_data