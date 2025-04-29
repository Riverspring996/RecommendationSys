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
#include <omp.h>   // 用于OpenMP并行
#include <tuple>   // 用于临时存储元素

using namespace std;

// 全局稀疏矩阵结构（CSR格式） - 仍然需要用于读取和映射
struct GlobalSparseMatrix {
    vector<double> values;
    vector<int> col_indices;
    vector<int> row_ptr;
    unordered_map<int, int> node_to_idx;
    unordered_map<int, int> idx_to_node;
    int n = 0;
    int nnz = 0;

    int getNodeId(int idx) const {
        auto it = idx_to_node.find(idx);
        return (it != idx_to_node.end()) ? it->second : -1;
    }
};

// --- BCSR (Blocked CSR) 实现 ---

// 代表一个块的 CSR 数据
struct CSRBlock {
    vector<double> values;      // 块内非零元素的值
    vector<int> col_indices;    // 块内非零元素的局部列索引 (0 to BLOCK_COL_SIZE-1)
    vector<int> row_ptr;        // 块内局部行指针 (大小 BLOCK_ROW_SIZE + 1)
    int nnz = 0;                // 块内非零元素数量

    CSRBlock(int block_height) : row_ptr(block_height + 1, 0) {} // 初始化行指针大小

    bool isEmpty() const { return nnz == 0; }
};

// 代表整个 BCSR 矩阵
class BlockedCSRMatrix {
public:
    // --- 块大小定义 ---
    // 块大小的选择对性能影响很大，需要根据缓存大小和数据特性调整
    // 常见的块大小是 4x4, 8x8, 16x16 等，但也可以是非方形的
    static const int BLOCK_ROW_SIZE = 512;
    static const int BLOCK_COL_SIZE = 512;
    // --------------------

    int global_n = 0;           // 全局矩阵维度
    int num_block_rows = 0;     // 行方向上的块数量
    int num_block_cols = 0;     // 列方向上的块数量
    vector<vector<CSRBlock>> blocks; // 存储所有块的 CSR 数据

    BlockedCSRMatrix(int n) : global_n(n) {
        if (n <= 0) return;
        num_block_rows = (n + BLOCK_ROW_SIZE - 1) / BLOCK_ROW_SIZE;
        num_block_cols = (n + BLOCK_COL_SIZE - 1) / BLOCK_COL_SIZE;
        // 初始化块结构，每个块都需要知道其高度来初始化 row_ptr
        blocks.resize(num_block_rows);
        for(int i = 0; i < num_block_rows; ++i) {
            blocks[i].reserve(num_block_cols); // 预分配空间
            int current_block_height = (i == num_block_rows - 1) ?
                                       (n - i * BLOCK_ROW_SIZE) : BLOCK_ROW_SIZE;
            for(int j = 0; j < num_block_cols; ++j) {
                blocks[i].emplace_back(current_block_height); // 使用块的实际高度
            }
        }
    }

    // 从全局 CSR 构建 BCSR 矩阵
    void buildFromGlobalCSR(const GlobalSparseMatrix& global_csr) {
        if (global_n == 0 || global_csr.n != global_n) return;

        // 1. 临时存储每个块的元素 (local_row, local_col, value)
        vector<vector<vector<tuple<int, int, double>>>> temp_blocks(
            num_block_rows, vector<vector<tuple<int, int, double>>>(num_block_cols)
        );

        for (int global_row = 0; global_row < global_n; ++global_row) {
            int block_row_idx = global_row / BLOCK_ROW_SIZE;
            int local_row = global_row % BLOCK_ROW_SIZE;

            int row_start = global_csr.row_ptr[global_row];
            int row_end = global_csr.row_ptr[global_row + 1];

            for (int k = row_start; k < row_end; ++k) {
                int global_col = global_csr.col_indices[k];
                double value = global_csr.values[k];

                int block_col_idx = global_col / BLOCK_COL_SIZE;
                int local_col = global_col % BLOCK_COL_SIZE;

                if (block_row_idx < num_block_rows && block_col_idx < num_block_cols) {
                    temp_blocks[block_row_idx][block_col_idx].emplace_back(local_row, local_col, value);
                }
            }
        }

        // 2. 为每个块构建局部 CSR
        #pragma omp parallel for collapse(2) // 并行构建块 CSR
        for (int bi = 0; bi < num_block_rows; ++bi) {
            for (int bj = 0; bj < num_block_cols; ++bj) {
                auto& elements = temp_blocks[bi][bj];
                if (elements.empty()) continue;

                CSRBlock& block = blocks[bi][bj];
                block.nnz = elements.size();
                block.values.resize(block.nnz);
                block.col_indices.resize(block.nnz);

                // 为了构建 CSR，需要按行排序块内元素
                sort(elements.begin(), elements.end());

                int current_nnz = 0;
                int current_local_row = -1;
                for (const auto& elem : elements) {
                    int local_row, local_col;
                    double value;
                    tie(local_row, local_col, value) = elem;

                    // 填充 row_ptr
                    // 当遇到新的一行时，更新前面所有行的指针直到当前行
                    if (local_row > current_local_row) {
                        for (int r = current_local_row + 1; r <= local_row; ++r) {
                            block.row_ptr[r] = current_nnz;
                        }
                        current_local_row = local_row;
                    }

                    block.values[current_nnz] = value;
                    block.col_indices[current_nnz] = local_col; // 存储局部列索引
                    current_nnz++;
                }
                // 填充最后一个元素的行以及之后所有空行的 row_ptr
                int block_height = block.row_ptr.size() - 1;
                for (int r = current_local_row + 1; r <= block_height; ++r) {
                     block.row_ptr[r] = current_nnz;
                }
            }
        }
    }

    // BCSR 矩阵向量乘法 y = A * x (A 是 this)
    void multiplyVector(const vector<double>& x, vector<double>& y) const {
        if (global_n == 0) return;
        fill(y.begin(), y.end(), 0.0);

        #pragma omp parallel for // 按块行并行
        for (int bi = 0; bi < num_block_rows; ++bi) {
            int global_row_start = bi * BLOCK_ROW_SIZE;

            for (int bj = 0; bj < num_block_cols; ++bj) {
                const CSRBlock& block = blocks[bi][bj];
                if (block.isEmpty()) continue;

                int global_col_start = bj * BLOCK_COL_SIZE;
                int block_height = block.row_ptr.size() - 1; // 获取当前块的实际高度

                // 执行块内 CSR SpMV
                for (int local_row = 0; local_row < block_height; ++local_row) {
                    double sum = 0.0;
                    int global_row = global_row_start + local_row;
                    if (global_row >= global_n) continue; // 防止越界 (最后一行块可能不满)

                    int row_nnz_start = block.row_ptr[local_row];
                    int row_nnz_end = block.row_ptr[local_row + 1];

                    for (int k = row_nnz_start; k < row_nnz_end; ++k) {
                        int local_col = block.col_indices[k];
                        double value = block.values[k];
                        int global_col = global_col_start + local_col;

                        if (global_col < global_n) { // 确保全局列索引有效
                            sum += value * x[global_col];
                        }
                    }
                    // --- 块内 CSR SpMV 结束 ---

                    // 原子地将块计算结果累加到全局 y 向量
                    //#pragma omp atomic
                    y[global_row] += sum;
                }
            }
        }
    }
};

// --- 读取和 PageRank 计算 ---

// 读取数据集并构建全局稀疏矩阵 (CSR格式)
GlobalSparseMatrix readGraph(const string& filename) {
    GlobalSparseMatrix graph;
    ifstream file(filename);
    if (!file) { /* ... error handling ... */ exit(1); }

    vector<pair<int, int>> edges;
    unordered_map<int, vector<int>> adj_list;
    int idx = 0;
    int from, to;

    // 第一次遍历: 读边, 建邻接表, 节点映射
    while (file >> from >> to) {
        edges.push_back({from, to});
        adj_list[from].push_back(to);
        if (graph.node_to_idx.find(from) == graph.node_to_idx.end()) {
            graph.node_to_idx[from] = idx; graph.idx_to_node[idx] = from; idx++;
        }
        if (graph.node_to_idx.find(to) == graph.node_to_idx.end()) {
            graph.node_to_idx[to] = idx; graph.idx_to_node[idx] = to; idx++;
        }
    }
    file.close();
    graph.n = idx;
    graph.nnz = edges.size();

    // 初始化全局 CSR 数组
    graph.values.resize(graph.nnz);
    graph.col_indices.resize(graph.nnz);
    graph.row_ptr.resize(graph.n + 1, 0);
    vector<int> out_degrees(graph.n, 0);
    vector<vector<pair<int, double>>> temp_rows(graph.n);

    // 计算出度
    for (const auto& pair : adj_list) {
        if (graph.node_to_idx.count(pair.first)) {
             out_degrees[graph.node_to_idx[pair.first]] = pair.second.size();
        }
    }

    // 构建 M^T (存储在全局 CSR 中)
    for (const auto& edge : edges) {
        int from_node = edge.first; int to_node = edge.second;
        if (graph.node_to_idx.count(from_node) && graph.node_to_idx.count(to_node)) {
            int from_idx = graph.node_to_idx[from_node];
            int to_idx = graph.node_to_idx[to_node];
            if (out_degrees[from_idx] > 0) {
                double prob = 1.0 / out_degrees[from_idx];
                temp_rows[to_idx].push_back({from_idx, prob}); // 行是 to_idx, 列是 from_idx
            }
        }
    }

    // 填充全局 CSR 数组
    int current_nnz = 0;
    for (int i = 0; i < graph.n; ++i) {
        graph.row_ptr[i] = current_nnz;
        // sort(temp_rows[i].begin(), temp_rows[i].end()); // 可选排序
        for (const auto& pair : temp_rows[i]) {
            if (current_nnz < graph.nnz) {
                 graph.col_indices[current_nnz] = pair.first;
                 graph.values[current_nnz] = pair.second;
                 current_nnz++;
            } else { /* error */ }
        }
    }
    graph.row_ptr[graph.n] = current_nnz;
    if (current_nnz < graph.nnz) { // 调整大小
        graph.nnz = current_nnz;
        graph.values.resize(graph.nnz);
        graph.col_indices.resize(graph.nnz);
    }
    return graph;
}

// 使用 BCSR 优化的 PageRank 算法
vector<pair<int, double>> computePageRank(const GlobalSparseMatrix& global_csr_graph, double alpha = 0.85, double epsilon = 1e-8) {
    int n = global_csr_graph.n;
    if (n == 0) return {};

    vector<double> pr(n, 1.0 / n);
    vector<double> next_pr(n, 0.0);
    vector<double> temp_pr(n, 0.0); // 用于存储 M^T * pr

    // 找出 Dead-ends (使用全局 CSR 的列索引)
    vector<bool> is_dead_end(n, true);
    for (int col_idx : global_csr_graph.col_indices) {
        if (col_idx >= 0 && col_idx < n) {
            is_dead_end[col_idx] = false;
        }
    }

    // --- 构建 BCSR 矩阵 ---
    auto start_build = chrono::high_resolution_clock::now();
    BlockedCSRMatrix bcsr_matrix(n);
    bcsr_matrix.buildFromGlobalCSR(global_csr_graph);
    auto end_build = chrono::high_resolution_clock::now();
    auto duration_build = chrono::duration_cast<chrono::milliseconds>(end_build - start_build).count();
    cout << "构建 BCSR 矩阵耗时: " << duration_build << " 毫秒" << endl;
    // -----------------------

    double teleport_prob = (1.0 - alpha) / n;
    int iterations = 0;
    double diff = 1.0;
    auto start_pagerank = chrono::high_resolution_clock::now();

    while (diff > epsilon && iterations < 100) {
        // --- BCSR 矩阵向量乘法 ---
        auto start_iter = chrono::high_resolution_clock::now();
        bcsr_matrix.multiplyVector(pr, temp_pr); // 使用 BCSR 乘法
        auto end_iter = chrono::high_resolution_clock::now();
        auto multi_duration = chrono::duration_cast<chrono::milliseconds>(end_iter - start_iter).count();
        // cout << " 迭代 " << iterations + 1 << " BCSR 乘法耗时: " << multi_duration << " 毫秒" << endl;
        // --------------------------

        // 处理 Dead-ends 和随机跳转 (可以并行优化)
        double dead_end_contribution = 0.0;
        #pragma omp parallel for reduction(+:dead_end_contribution)
        for (int i = 0; i < n; i++) {
            if (is_dead_end[i]) {
                dead_end_contribution += pr[i];
            }
        }
        dead_end_contribution *= alpha / n;

        // 计算 next_pr (可以并行优化)
        #pragma omp parallel for simd
        for (int i = 0; i < n; i++) {
            next_pr[i] = alpha * temp_pr[i] + teleport_prob + dead_end_contribution;
        }

        // 计算差异 (可以并行优化)
        diff = 0.0;
        #pragma omp parallel for simd reduction(+:diff)
        for (int i = 0; i < n; i++) {
            diff += abs(next_pr[i] - pr[i]);
        }

        pr.swap(next_pr); // 更新 PR 值
        iterations++;
    }

    cout << "PageRank 计算收敛，迭代次数: " << iterations << endl;
    auto end_pagerank = chrono::high_resolution_clock::now();
    auto duration_pagerank = chrono::duration_cast<chrono::milliseconds>(end_pagerank - start_pagerank).count();
    cout << "PageRank 计算 (BCSR) 耗时: " << duration_pagerank << " 毫秒" << endl;

    // 结果关联和排序
    vector<pair<int, double>> result;
    result.reserve(n);
    for (int i = 0; i < n; i++) {
        result.push_back({global_csr_graph.getNodeId(i), pr[i]});
    }
    sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    return result;
}

int main() {
    SetConsoleOutputCP(65001);
    int num_threads = 8; // 设置线程数
    omp_set_num_threads(num_threads);
    cout << "使用 OpenMP 线程数: " << num_threads << endl;

    auto start_time = chrono::high_resolution_clock::now();

    // 读取数据并构建全局 CSR 图
    GlobalSparseMatrix graph = readGraph("Data.txt");
    cout << "图数据读取完成 (全局 CSR)，节点数: " << graph.n << "，非零元素数: " << graph.nnz << endl;

    // 计算 PageRank (使用 BCSR 优化)
    auto pagerank_result = computePageRank(graph);

    // 输出结果
    ofstream outfile("Res_bcsr.txt");
    if (!outfile) { /* ... error handling ... */ return 1; }
    int top_k = min(100, static_cast<int>(pagerank_result.size()));
    outfile << fixed << setprecision(8);
    for (int i = 0; i < top_k; i++) {
        if (pagerank_result[i].first != -1) {
             outfile << pagerank_result[i].first << " " << pagerank_result[i].second << "\n";
        }
    }
    outfile.close();

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

    cout << "计算完成，结果已保存到 Res_bcsr.txt" << endl;
    cout << "总运行时间: " << duration << " 毫秒" << endl;
    system("pause");
    // 编译命令: g++ pagerank_bcsr.cpp -o pagerank_bcsr.exe -fopenmp -O2 (或 -O3)

    return 0;
}