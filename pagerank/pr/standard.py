import networkx as nx

# Script: compare_pagerank.py
# 功能：
# 1. 使用 NetworkX 计算参考 PageRank，并输出 nx_Res.txt。
# 2. 加载用户结果 Res.txt 和参考结果 nx_Res.txt。
# 3. 计算 Top-100 准确率。
# 4. 列出在参考中但不在用户结果中的节点（only_in_ref）。
# 5. 列出在用户结果中但不在参考中的节点（only_in_user）。
# 6. 列出每个位置上不一致的节点对（positional_mismatches）。

def compute_reference(input_file='Data.txt', output_file='nx_Res.txt', alpha=0.85, topk=100):
    G = nx.DiGraph()
    with open(input_file, 'r') as f:
        for line in f:
            u, v = line.split()
            G.add_edge(int(u), int(v))
    pr = nx.pagerank(G, alpha=alpha)
    top_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:topk]
    with open(output_file, 'w') as fw:
        for node, score in top_nodes:
            fw.write(f"{node} {score:.8f}\n")
    print(f"Reference PageRank result written to {output_file}.")

def compare_results(file_user='Respy.txt', file_ref='block.txt', topk=100):
    def load_nodes(path):
        with open(path, 'r') as f:
            return [int(line.split()[0]) for line in f if line.strip()]

    user_nodes = load_nodes(file_user)[:topk]
    ref_nodes = load_nodes(file_ref)[:topk]

    set_user = set(user_nodes)
    set_ref = set(ref_nodes)

    intersection = set_user & set_ref
    accuracy = len(intersection) / topk * 100

    only_in_ref = [n for n in ref_nodes if n not in set_user]
    only_in_user = [n for n in user_nodes if n not in set_ref]
    positional_mismatches = [
        (i+1, user_nodes[i], ref_nodes[i])
        for i in range(topk) if user_nodes[i] != ref_nodes[i]
    ]

    print(f"Top-{topk} intersection count: {len(intersection)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("\nNodes only in reference (但不在用户结果中):")
    print(only_in_ref)
    print("\nNodes only in user result (但不在参考结果中):")
    print(only_in_user)
    print("\nPositional mismatches (位置, 用户节点, 参考节点):")
    for pos, u, r in positional_mismatches:
        print(f"Position {pos}: User {u} vs Reference {r}")

if __name__ == "__main__":
    # 1. 生成参考答案
    # compute_reference()
    # 2. 对比两份结果并输出详细差异
    compare_results()

# 使用：
#   python3 compare_pagerank.py
# 确保当前目录包含 Data.txt, Res.txt。

