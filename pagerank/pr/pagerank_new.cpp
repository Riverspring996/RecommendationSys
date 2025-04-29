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
#include <omp.h> // 用于OpenMP并行

using namespace std;

// 稀疏矩阵结构（CSR格式）
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
class BlockMatrix {
public:
    static const int BLOCK_SIZE = 128;  // 较小的块大小，考虑到缓存友好
    
    BlockMatrix(int size) : n(size) {
        num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        blocks.resize(num_blocks * num_blocks);
    }
    
    // 从稀疏矩阵构建块矩阵
    void buildFromSparse(const SparseMatrix& sparse) {
        n = sparse.n;
        num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        blocks.clear();
        blocks.resize(num_blocks * num_blocks);
        
        // 将稀疏矩阵元素填入对应的块
        for (size_t i = 0; i < sparse.values.size(); i++) {
            int row = sparse.rows[i];
            int col = sparse.cols[i];
            double val = sparse.values[i];
            set(row, col, val);
        }
    }
    
    // 块矩阵向量乘法, y = Ax
    void multiplyVector(const vector<double>& x, vector<double>& y) const {
        fill(y.begin(), y.end(), 0.0);
        
        // 使用OpenMP进行并行化
        #pragma omp parallel for collapse(2)
        for (int bi = 0; bi < num_blocks; bi++) {
            for (int bj = 0; bj < num_blocks; bj++) {
                int block_idx = bi * num_blocks + bj;
                
                // 跳过空块
                if (blocks[block_idx].empty()) continue;
                
                const auto& block = blocks[block_idx];
                
                // 确定当前块的实际维度
                int row_start = bi * BLOCK_SIZE;
                int row_end = min(row_start + BLOCK_SIZE, n);
                int col_start = bj * BLOCK_SIZE;
                int col_end = min(col_start + BLOCK_SIZE, n);
                
                // 对块内每个元素进行矩阵-向量乘法
                for (int i = row_start; i < row_end; i++) {
                    for (int j = col_start; j < col_end; j++) {
                        int local_i = i - row_start;
                        int local_j = j - col_start;
                        int idx = local_i * BLOCK_SIZE + local_j;
                        
                        if (idx < block.size() && block[idx] != 0.0) {
                            y[i] += block[idx] * x[j];  // 向量与矩阵乘法
                        }
                    }
                }
            }
        }
    }
   
    // 获取某一块
    vector<double>& getBlock(int i, int j) {
        int idx = i * num_blocks + j;
        if (blocks[idx].empty()) {
            blocks[idx].resize(BLOCK_SIZE * BLOCK_SIZE, 0.0);
        }
        return blocks[idx];
    }
    
    // 获取矩阵中的元素
    double get(int i, int j) const {
        int block_i = i / BLOCK_SIZE;
        int block_j = j / BLOCK_SIZE;
        int local_i = i % BLOCK_SIZE;
        int local_j = j % BLOCK_SIZE;
        
        int block_idx = block_i * num_blocks + block_j;
        
        if (blocks[block_idx].empty()) return 0.0;
        
        const auto& block = blocks[block_idx];
        int idx = local_i * BLOCK_SIZE + local_j;
        
        if (idx >= block.size()) return 0.0;
        
        return block[idx];
    }
    
    // 设置矩阵中的元素
    void set(int i, int j, double value) {
        int block_i = i / BLOCK_SIZE;
        int block_j = j / BLOCK_SIZE;
        int local_i = i % BLOCK_SIZE;
        int local_j = j % BLOCK_SIZE;
        
        vector<double>& block = getBlock(block_i, block_j);
        block[local_i * BLOCK_SIZE + local_j] = value;
    }
    
private:
    int n;  // 矩阵维度
    int num_blocks;  // 每维的块数
    vector<vector<double>> blocks;  // 存储块的向量
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
    
    // 读取数据并构建邻接表
    while (file >> from >> to) {
        adj_list[from].push_back(to);
        if (graph.node_to_idx.find(from) == graph.node_to_idx.end()) {
            graph.node_to_idx[from] = idx;
            graph.idx_to_node[idx] = from;
            idx++;
        }
        if (graph.node_to_idx.find(to) == graph.node_to_idx.end()) {
            graph.node_to_idx[to] = idx;
            graph.idx_to_node[idx] = to;
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

// 使用块矩阵优化的PageRank算法
vector<pair<int, double>> computePageRank(const SparseMatrix& sparse_graph, double alpha = 0.85, double epsilon = 1e-8) {
    int n = sparse_graph.n;
    vector<double> pr(n, 1.0 / n);  // 初始PageRank值
    vector<double> next_pr(n, 0.0);  // 下一轮的PageRank值
    
    // 找出所有无出链的节点（Dead-ends）
    vector<bool> is_dead_end(n, true);
    for (int col : sparse_graph.cols) {
        is_dead_end[col] = false;
    }
    
    // 构建块矩阵
    BlockMatrix block_matrix(n);
    block_matrix.buildFromSparse(sparse_graph);
    
    double teleport_prob = (1.0 - alpha) / n;
    int iterations = 0;
    double diff = 1.0;
    
    // 迭代计算PageRank直到收敛
    while (diff > epsilon && iterations < 100) {
        // 矩阵-向量乘法 (使用块矩阵优化)
        vector<double> temp_pr(n, 0.0);
        block_matrix.multiplyVector(pr, temp_pr);
        
        // 处理Dead-ends（无出链节点）
        double dead_end_score = 0.0;
        for (int i = 0; i < n; i++) {
            if (is_dead_end[i]) {
                dead_end_score += alpha * pr[i] / n;
            }
            // 应用alpha系数到temp_pr
            temp_pr[i] *= alpha;
        }
        
        // 加上随机跳转和Dead-ends的贡献
        for (int i = 0; i < n; i++) {
            next_pr[i] = temp_pr[i] + teleport_prob + dead_end_score;
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
    
    // 将PageRank值与原始节点ID关联
    vector<pair<int, double>> result;
    for (int i = 0; i < n; i++) {
        result.push_back({sparse_graph.getNodeId(i), pr[i]});
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
    omp_set_num_threads(10);
    auto start_time = chrono::high_resolution_clock::now();
    
    // 读取数据并构建图
    SparseMatrix graph = readGraph("Data.txt");
    cout << "图数据读取完成，节点数: " << graph.n << "，边数: " << graph.values.size() << endl;
    
    // 计算PageRank (使用块矩阵和稀疏矩阵)
    auto pagerank_result = computePageRank(graph);
    
    // 输出前100个结果到文件
    ofstream outfile("Res_new.txt");
    if (!outfile) {
        cerr << "无法创建输出文件!" << endl;
        return 1;
    }
    
    int top_k = min(100, static_cast<int>(pagerank_result.size()));
    for (int i = 0; i < top_k; i++) {
        outfile << pagerank_result[i].first << " " << fixed << setprecision(8) << pagerank_result[i].second << "\n";
    }
    outfile.close();
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

    float duration_t = duration/1000;
    
    cout << "计算完成，结果已保存到Res.txt" << endl;
    cout << "总运行时间: " << duration << " 毫秒" << endl;
    system("pause");
    // 运行命令: g++ pagerank_new.cpp -o pagerank_new.exe -fopenmp
    
    return 0;
}
