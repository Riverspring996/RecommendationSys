#!/usr/bin/env python3
# coding: utf-8
import sys
import networkx as nx


def load_graph(path: str) -> nx.DiGraph:
    G = nx.DiGraph()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            src, dst = line.split()
            G.add_edge(int(src), int(dst))
    return G


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) >= 2 else "Data.txt"
    alpha = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.85

    print(f"读取 {data_path}，alpha={alpha}")
    G = load_graph(data_path)

    print("开始计算 PageRank …")
    pr = nx.pagerank(G, alpha=alpha, tol=1e-8)  # tol 越小收敛越严格

    top100 = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:100]

    with open("nbw.txt", "w", encoding="utf-8") as f:
        for node, score in top100:
            f.write(f"{node} {score:.10f}\n")

    print("已写入 nbw.txt")
