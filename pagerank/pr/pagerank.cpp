#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <cmath>
#include <numeric>
#include <chrono>
#include <windows.h>
#include <iomanip> // 用于setprecision

using namespace std;

// 用于存储图的稀疏矩阵表示
struct SparseMatrix {
    vector<int> rows;             // 非零元素的行索引
    vector<int> cols;             // 非零元素的列索引
    vector<double> values;        // 非零元素的值
    unordered_map<int, int> node_to_idx;  // 节点ID到索引的映射
    unordered_map<int, int> idx_to_node;  // 索引到节点ID的映射
    int n = 0;                          // 矩阵大小
    
    // 安全获取idx_to_node值的方法
    int getNodeId(int idx) const {
        auto it = idx_to_node.find(idx);
        if (it != idx_to_node.end()) {
            return it->second;
        }
        return -1; // 或其他适当的错误值
    }
};

// 块矩阵结构
struct BlockMatrix {
    static const int BLOCK_SIZE = 1024;  // 块大小
    vector<vector<double>> blocks;
    int n = 0;
    int num_blocks = 0;
    
    BlockMatrix(int size) {
        n = size;
        num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        blocks.resize(num_blocks * num_blocks);
    }
    
    // 获取某一块
    vector<double>& getBlock(int i, int j) {
        int idx = i * num_blocks + j;
        if (blocks[idx].empty()) {
            blocks[idx].resize(BLOCK_SIZE * BLOCK_SIZE, 0.0);
        }
        return blocks[idx];
    }
    
    // 获取矩阵中的一个元素
    double get(int i, int j) {
        int block_i = i / BLOCK_SIZE;
        int block_j = j / BLOCK_SIZE;
        int local_i = i % BLOCK_SIZE;
        int local_j = j % BLOCK_SIZE;
        
        vector<double>& block = getBlock(block_i, block_j);
        return block[local_i * BLOCK_SIZE + local_j];
    }
    
    // 设置矩阵中的一个元素
    void set(int i, int j, double value) {
        int block_i = i / BLOCK_SIZE;
        int block_j = j / BLOCK_SIZE;
        int local_i = i % BLOCK_SIZE;
        int local_j = j % BLOCK_SIZE;
        
        vector<double>& block = getBlock(block_i, block_j);
        block[local_i * BLOCK_SIZE + local_j] = value;
    }
};

// 读取数据集并构建稀疏矩阵
SparseMatrix readGraph(const string& filename) {
    SparseMatrix graph;
    ifstream file(filename);
    
    if (!file) {
        cerr << "无法打开文件: " << filename << endl;
        exit(1);
    }
    
    int from, to;
    int idx = 0;
    unordered_map<int, vector<int>> adj_list;
    vector<int> all_nodes;
    
    // 读取数据并构建邻接表
    while (file >> from >> to) {
        adj_list[from].push_back(to);
        if (graph.node_to_idx.find(from) == graph.node_to_idx.end()) {
            graph.node_to_idx[from] = idx;
            graph.idx_to_node[idx] = from;
            all_nodes.push_back(from);
            idx++;
        }
        if (graph.node_to_idx.find(to) == graph.node_to_idx.end()) {
            graph.node_to_idx[to] = idx;
            graph.idx_to_node[idx] = to;
            all_nodes.push_back(to);
            idx++;
        }
    }
    
    graph.n = idx;
    vector<int> out_degrees(graph.n, 0);
    
    // 计算每个节点的出度
    for (const auto& pair : adj_list) {
        int node_idx = graph.node_to_idx[pair.first];
        out_degrees[node_idx] = pair.second.size();
    }
    
    // 构建稀疏转移矩阵
    for (const auto& pair : adj_list) {
        int from_idx = graph.node_to_idx[pair.first];
        double prob = 1.0 / out_degrees[from_idx];
        
        for (int to_node : pair.second) {
            int to_idx = graph.node_to_idx[to_node];
            graph.rows.push_back(to_idx);
            graph.cols.push_back(from_idx);
            graph.values.push_back(prob);
        }
    }
    
    return graph;
}

// 执行PageRank算法
vector<pair<int, double>> computePageRank(const SparseMatrix& graph, double alpha = 0.85, double epsilon = 1e-8) {
    int n = graph.n;
    vector<double> pr(n, 1.0 / n);  // 初始PageRank值
    vector<double> next_pr(n, 0.0);  // 下一轮的PageRank值
    
    // 找出所有无出链的节点（Dead-ends）
    vector<bool> is_dead_end(n, true);
    for (int col : graph.cols) {
        is_dead_end[col] = false;
    }

    auto start_time = chrono::high_resolution_clock::now();

    double teleport_prob = (1.0 - alpha) / n;
    int iterations = 0;
    double diff = 1.0;
    
    // 迭代计算PageRank直到收敛
    while (diff > epsilon && iterations < 100) {
        // 重置下一轮的PR值
        fill(next_pr.begin(), next_pr.end(), 0.0);
        
        auto start_time1 = chrono::high_resolution_clock::now();

        // PageRank矩阵向量乘法（稀疏矩阵优化）
        for (size_t i = 0; i < graph.values.size(); i++) {
            int row = graph.rows[i];
            int col = graph.cols[i];
            double val = graph.values[i];
            next_pr[row] += alpha * val * pr[col];
        }

        auto end_time1 = chrono::high_resolution_clock::now();
        auto multi_duration = chrono::duration_cast<chrono::milliseconds>(end_time1 - start_time1).count();
        cout << "矩阵-向量乘法耗时: " << multi_duration << " 毫秒" << endl;
        
        // 处理Dead-ends（无出链节点）
        double dead_end_score = 0.0;
        for (int i = 0; i < n; i++) {
            if (is_dead_end[i]) {
                dead_end_score += alpha * pr[i] / n;
            }
        }
        
        // 加上随机跳转和Dead-ends的贡献
        for (int i = 0; i < n; i++) {
            next_pr[i] += teleport_prob + dead_end_score;
        }
        
        // 计算新旧PageRank的差异
        diff = 0.0;
        for (int i = 0; i < n; i++) {
            diff += abs(next_pr[i] - pr[i]);
        }
        
        // 更新PageRank值
        pr = next_pr;
        iterations++;
    }
    
    cout << "PageRank计算收敛，迭代次数: " << iterations << endl;
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    cout << "PageRank计算耗时: " << duration << " 毫秒" << endl;
    
    // 将PageRank值与原始节点ID关联
    vector<pair<int, double>> result;
    for (int i = 0; i < n; i++) {
        result.push_back({graph.getNodeId(i), pr[i]});
    }
    
    // 按PageRank值降序排序
    sort(result.begin(), result.end(),
         [](const pair<int, double>& a, const pair<int, double>& b) { 
             return a.second > b.second; 
         });
    
    return result;
}

int main() {
    SetConsoleOutputCP(65001);
    auto start_time = chrono::high_resolution_clock::now();
    
    // 读取数据并构建图
    SparseMatrix graph = readGraph("Data.txt");
    cout << "图数据读取完成，节点数: " << graph.n << "，边数: " << graph.values.size() << endl;
    
    // 计算PageRank
    auto pagerank_result = computePageRank(graph);
    
    // 输出前100个结果到文件
    ofstream outfile("Res.txt");
    if (!outfile) {
        cerr << "无法创建输出文件!" << endl;
        return 1;
    }
    
    int top_k = min(100, static_cast<int>(pagerank_result.size()));
    for (int i = 0; i < top_k; i++) {
        outfile << pagerank_result[i].first << " " << fixed << setprecision(10) << pagerank_result[i].second << "\n";
    }
    outfile.close();
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

    float duration_t = duration/1000;
    
    cout << "计算完成，结果已保存到Res.txt" << endl;
    cout << "总运行时间: " << duration << " 毫秒" << endl;
    system("pause");
    
    return 0;
}