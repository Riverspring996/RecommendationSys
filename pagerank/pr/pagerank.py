import numpy as np
import time
from scipy import sparse
import sys
import gc

def load_graph(file_path):
    """加载数据集并构建图，使用更高效的内存表示"""
    print("开始加载数据...")
    
    # 先遍历一次文件，获取节点数量和边数量的估计
    max_node_id = -1
    edge_count = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                from_node, to_node = int(parts[0]), int(parts[1])
                max_node_id = max(max_node_id, from_node, to_node)
                edge_count += 1
    
    # 使用集合跟踪实际存在的节点
    node_set = set()
    # 使用邻接表存储图
    out_links_temp = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                from_node, to_node = int(parts[0]), int(parts[1])
                node_set.add(from_node)
                node_set.add(to_node)
                
                if from_node not in out_links_temp:
                    out_links_temp[from_node] = []
                out_links_temp[from_node].append(to_node)
    
    # 创建节点映射
    sorted_nodes = sorted(node_set)
    node_to_idx = {node: idx for idx, node in enumerate(sorted_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    n = len(node_set)
    
    # 创建邻接表（使用列表的列表更节省内存）
    out_links = [[] for _ in range(n)]
    for src, dst_list in out_links_temp.items():
        src_idx = node_to_idx[src]
        for dst in dst_list:
            dst_idx = node_to_idx[dst]
            out_links[src_idx].append(dst_idx)
    
    # 释放临时内存
    del out_links_temp
    gc.collect()
    
    print(f"数据加载完成，共有 {n} 个节点和 {edge_count} 条边")
    return out_links, node_to_idx, idx_to_node, n

def compute_pagerank_iterative(out_links, n, alpha=0.85, max_iter=100, tol=1e-6):
    """使用迭代方法计算PageRank，不显式构建转移矩阵，节省内存"""
    print(f"开始计算PageRank (alpha={alpha})...")
    
    # 计算节点出度
    out_degrees = np.zeros(n, dtype=np.int32)
    for i, links in enumerate(out_links):
        out_degrees[i] = len(links)
    
    # 初始化PageRank向量
    v = np.ones(n, dtype=np.float64) / n
    v_new = np.zeros(n, dtype=np.float64)
    
    # 迭代计算
    for iteration in range(max_iter):
        v_new.fill(0)
        
        # 计算dead ends的贡献
        dead_end_sum = 0
        for i in range(n):
            if out_degrees[i] == 0:
                dead_end_sum += v[i]
        
        # 为每个节点添加dead ends的贡献
        dead_end_contrib = alpha * dead_end_sum / n
        v_new.fill(dead_end_contrib)
        
        # 计算普通节点的贡献
        for i in range(n):
            if out_degrees[i] > 0:
                val = v[i] / out_degrees[i] * alpha
                for j in out_links[i]:
                    v_new[j] += val
        
        # 添加随机跳转
        teleport = (1 - alpha) / n
        v_new += teleport
        
        # 检查收敛性
        err = np.sum(np.abs(v_new - v))
        v, v_new = v_new, v  # 交换引用以避免复制
        
        # 归一化
        v_sum = np.sum(v)
        if v_sum > 0:
            v /= v_sum
        
        print(f"迭代 {iteration+1}: 误差 = {err:.10f}")
        if err < tol:
            print(f"PageRank算法在第 {iteration+1} 次迭代后收敛")
            break
    
    return v

def main():
    start_time = time.time()
    
    # 检查是否有命令行参数指定数据文件
    file_path = "Data.txt"
    
    # 加载数据
    out_links, node_to_idx, idx_to_node, n = load_graph(file_path)
    
    # 计算PageRank
    alpha = 0.85  # 跳转概率
    scores = compute_pagerank_iterative(out_links, n, alpha)
    
    # 获取排名前100的节点
    top_indices = np.argsort(-scores)[:100]
    
    # 输出结果
    with open("Respy.txt", "w") as f:
        for idx in top_indices:
            node_id = idx_to_node[idx]
            score = scores[idx]
            f.write(f"{node_id} {score:.16f}\n")
    
    elapsed_time = time.time() - start_time
    memory_mb = get_memory_usage()
    
    print(f"总计用时: {elapsed_time:.2f} 秒")
    print(f"最大内存使用: {memory_mb:.2f} MB")
    print(f"结果已保存到 Res.txt")

def get_memory_usage():
    """获取当前进程的内存使用情况（MB）"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0  # 如果无法导入psutil，返回0

if __name__ == "__main__":
    main()