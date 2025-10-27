#include "h2h.h"
#include "spf.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include "EmbeddingService.h"

int nofcats, K;
vector<vector<double>> categoryEmbeddings; //the first one is empty
vector<string> catname; //the first one is empty
void readCategoryID(string sfile) {
    string prefix = string("../data/") + sfile + string("/");
    string categoryIDFile = prefix + string("Categories_ID.txt");
    
    ifstream file(categoryIDFile);    
    string line;
    nofcats = 0;
    catname.push_back(string(""));
    while (getline(file, line)) {
        if (line.empty()) continue;  
        int pos = line.find(' ');
        stringstream ss(line);
        int id;
        ss >> id;
        catname.push_back(line.substr(pos + 1));
        nofcats++;
    }
    cout << "nofcats " << nofcats << endl;
}

void readPOINodeMapping(string sfile){    
    string prefix = string("../data/") + sfile + string("/");
    ifstream in(prefix + string("poi_node_mapping.txt"));
    string line;
    int node = -1;
    set<int> cats, allcats;
    map<int, set<int>> categoryToNodes;
    
    // 读取文件构建数据结构
    while (getline(in, line)) {
        if (line.find('-') != string::npos) {
            if (node != -1) {
                for (int c : cats) {
                    categoryToNodes[c].insert(node-1);
                }
            }
            node = stoi(line.substr(0, line.find('-')));
            cats.clear();
        } else {
            int pos = line.find('<');
            if (pos != string::npos) {
                stringstream ss(line.substr(pos + 1));
                string cat;
                while (getline(ss, cat, '^')) {
                    if (!cat.empty()){
                        cats.insert(stoi(cat));
                        allcats.insert(stoi(cat));
                    }
                }
            }
        }
    }
    
    if (node != -1) {
        for (int c : cats) {
            categoryToNodes[c].insert(node-1);
        }
    }
    for (int c : allcats){
        for(int j: categoryToNodes[c]){
            category2Nodes[c].push_back(j);
        }
    }
    in.close();
}

// 读取category embeddings文件
void readCategoryEmbeddings(string sfile) {
    string prefix = string("../data/") + sfile + string("/");
    string categoryEmbeddingFile = prefix + string("category_embeddings.txt");
    
    ifstream file(categoryEmbeddingFile);    
    categoryEmbeddings.clear();
    vector<double> emb1;
    categoryEmbeddings.push_back(emb1);
    string line;

    while (getline(file, line)) {
        if (line.empty()) continue;  
        vector<double> embedding;
        stringstream ss(line);
        double value;
        while (ss >> value) {
            embedding.push_back(value);
        }
        if (!embedding.empty()) {
            categoryEmbeddings.push_back(embedding);
        }
    }
    
    file.close();
    cout << "Loaded " << categoryEmbeddings.size() - 1 << " category embeddings, dimension: " 
         << (categoryEmbeddings.empty() ? 0 : categoryEmbeddings[1].size()) << endl;
}


// 计算两个归一化向量的点积（余弦相似度）
double dotProduct(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size()) return -1.0;
    
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

void findSimilarCategories(const vector<double>& queryEmbedding, double alpha, vector<int>& result) {
    result.clear();
    double top1 = -1;
    for (size_t i = 1; i <= categoryEmbeddings.size() - 1; ++i) {
        double similarity = dotProduct(queryEmbedding, categoryEmbeddings[i]);
        top1 = max(top1, similarity);
    }
    //cout << top1 << " " << alpha << " " << categoryEmbeddings.size() - 1 << endl;
    for (size_t i = 1; i <= categoryEmbeddings.size() - 1; ++i) {
        double similarity = dotProduct(queryEmbedding, categoryEmbeddings[i]);
        //cout << catname[i] << "---" << similarity << endl;
        if (similarity >= top1 - alpha) {
            cout << catname[i] << " " << similarity << endl;
            result.push_back((int)(i));
        }
    }   
}

vector<vector<int>> Vr;
vector<int> landmarks;  // landmark节点集合
map<pair<int,int>, double> landmarkDistances; // 所有节点到landmark的距离
map<pair<int,int>, double> exactDistances; // 已计算的精确距离
const double INF = DBL_MAX;
void selectLandmarksGreedyMerge(int numLandmarks) {
    landmarks.clear();
    
    // 收集所有POI节点
    set<int> allPOIs;
    for (auto& kv : category2Nodes) {
        for (int node : kv.second) {
            allPOIs.insert(node);
        }
    }
    
    vector<int> candidateNodes(allPOIs.begin(), allPOIs.end());
    
    if (candidateNodes.size() <= numLandmarks) {
        landmarks = candidateNodes;
        cout << "Selected all " << landmarks.size() << " POI nodes as landmarks" << endl;
        return;
    }
    
    // 简化策略：基于空间分布选择landmark
    // 1. 随机选择第一个landmark
    srand(time(nullptr));
    landmarks.push_back(candidateNodes[rand() % candidateNodes.size()]);
    
    // 2. 迭代选择距离已选landmarks最远的节点
    for (int iter = 1; iter < numLandmarks; iter++) {
        int bestNode = -1;
        double maxMinDistance = -1;
        
        for (int candidate : candidateNodes) {
            // 跳过已选择的节点
            if (find(landmarks.begin(), landmarks.end(), candidate) != landmarks.end()) 
                continue;
            
            // 计算candidate到所有已选landmarks的最小距离
            double minDistToLandmarks = INF;
            for (int landmark : landmarks) {
                double dist = h2hQuery(candidate, landmark);
                minDistToLandmarks = min(minDistToLandmarks, dist);
            }
            
            // 选择最小距离最大的节点（最远的节点）
            if (minDistToLandmarks > maxMinDistance) {
                maxMinDistance = minDistToLandmarks;
                bestNode = candidate;
            }
        }
        
        if (bestNode != -1) {
            landmarks.push_back(bestNode);
        }
    }
    
    cout << "Selected " << landmarks.size() << " landmarks using simplified GM strategy" << endl;
}


// 2. 构建RNII索引 - 按照论文Section 5.2
void buildRNII(int numLandmarks = 50) {
    selectLandmarksGreedyMerge(numLandmarks);
    
    landmarkDistances.clear();
    set<int> allNodes;
    
    // 收集所有相关节点
    for (auto& kv : category2Nodes) {
        for (int node : kv.second) {
            allNodes.insert(node);
        }
    }
    
    // 预计算所有节点到landmarks的距离
    for (int landmark : landmarks) {
        for (int node : allNodes) {
            pair<int,int> key = {min(node, landmark), max(node, landmark)};
            double dist = h2hQuery(node, landmark);
            landmarkDistances[key] = dist;
        }
    }
    
    cout << "RNII built with " << landmarkDistances.size() << " distances" << endl;
}
double getLowerBoundDistance(int u, int v) {
    if (u == v) return 0.0;
    
    // 首先检查是否已有精确距离
    pair<int,int> key = {min(u,v), max(u,v)};
    if (exactDistances.count(key)) {
        return exactDistances[key];
    }
    
    // 按照论文公式：LB_{pi,pj} = max_l(|V_{pi}[l] - V_{pj}[l]|)
    double maxLowerBound = 0.0;
    
    for (int landmark : landmarks) {
        // 获取距离向量元素
        auto getDistToLandmark = [&](int node) -> double {
            pair<int,int> lkey = {min(node, landmark), max(node, landmark)};
            if (landmarkDistances.count(lkey)) {
                return landmarkDistances[lkey];
            }
            // 动态计算（主要用于起点终点）
            double dist = h2hQuery(node, landmark);
            landmarkDistances[lkey] = dist;
            return dist;
        };
        
        double distUL = getDistToLandmark(u);
        double distVL = getDistToLandmark(v);
        maxLowerBound = max(maxLowerBound, abs(distUL - distVL));
    }
    
    return maxLowerBound;
}

// 4. OSE算法 - 严格按照Definition 6的动态规划公式
void OSE1(int vs, int vd, double costthresh, vector<int> &optimalPath, double &minTotalCost, double wub) {
    int k = Vr.size();
    
    // 动态规划矩阵：OS[i][j]表示到第i个类别第j个POI的最优成本
    vector<map<int, double>> OS(k + 1);
    vector<map<int, int>> parent(k + 1);
    
    // 初始化：OS[0] = 0 (从起点开始)
    OS[0][vs] = 0.0;
    
    // 按照公式填充DP表
    for (int i = 1; i <= k; i++) {
        // 对每个第i个类别的POI
        for (int j : Vr[i-1]) {
            OS[i][j] = INF;
            parent[i][j] = -1;
            
            // 从前一层的所有状态转移
            // OS[i,j] = min{OS[i-1,l] + dist(p^{c_{i-1}}_l, p^{c_i}_j)}
            for (auto& [l, cost_l] : OS[i-1]) {
                if (cost_l >= costthresh) continue; // 剪枝
                if (i > 1){
                    if (cost_l + h2hQuery(l, vd) > wub) {
                        continue;
                    }
                }
                double edgeCost = getLowerBoundDistance(l, j);
                double newCost = cost_l + edgeCost;
                
                if (newCost < OS[i][j] && newCost < costthresh) {
                    OS[i][j] = newCost;
                    parent[i][j] = l;
                }
            }
        }
    }
    
    // 找到连接终点的最优路径
    minTotalCost = INF;
    int bestLastPOI = -1;
    
    for (auto& [lastPOI, costToLast] : OS[k]) {
        if (costToLast < INF) {
            double totalCost = costToLast + getLowerBoundDistance(lastPOI, vd);
            if (totalCost < minTotalCost) {
                minTotalCost = totalCost;
                bestLastPOI = lastPOI;
            }
        }
    }
    
    // 重构路径
    int curr = bestLastPOI;
    
    for (int i = k; i >= 1; i--) {
        optimalPath.push_back(curr);
        curr = parent[i][curr];
    }
    
    reverse(optimalPath.begin(), optimalPath.end());
}


void setExactDistance(int u, int v, double dist) {
    pair<int,int> key = {min(u,v), max(u,v)};
    exactDistances[key] = dist;
}

// 5. computeRouteDistance - 严格按照Algorithm 1 Line 6
double computeRouteDistance1(int vs, const vector<int>& Rguide, int vd) {
    if (Rguide.empty()) return INF;
    
    double totalDistance = 0.0;
    
    // 计算并缓存所有路径段的精确距离
    // vs -> Rguide[0]
    double d1 = h2hQuery(vs, Rguide[0]);
    setExactDistance(vs, Rguide[0], d1);
    totalDistance += d1;
    
    // Rguide[i-1] -> Rguide[i]
    for (int i = 1; i < Rguide.size(); i++) {
        double d = h2hQuery(Rguide[i-1], Rguide[i]);
        setExactDistance(Rguide[i-1], Rguide[i], d);
        totalDistance += d;
    }
    
    // Rguide[last] -> vd
    double d2 = h2hQuery(Rguide.back(), vd);
    setExactDistance(Rguide.back(), vd, d2);
    totalDistance += d2;
    
    return totalDistance;
}

// 优化后的OSE：直接计算精确距离，不再使用下界
void OSE(int vs, int vd, double costthresh, vector<int> &optimalPath, double &minTotalCost, double wub) {
    int k = Vr.size();
    
    vector<map<int, double>> OS(k + 1);
    vector<map<int, int>> parent(k + 1);
    
    OS[0][vs] = 0.0;
    
    // 辅助函数：获取或计算精确距离
    auto getExactDistance = [](int u, int v) -> double {
        if (u == v) return 0.0;
        pair<int,int> key = {min(u, v), max(u, v)};
        
        if (exactDistances.count(key)) {
            return exactDistances[key];
        }
        
        double dist = h2hQuery(u, v);
        exactDistances[key] = dist;
        return dist;
    };
    
    // DP主循环
    for (int i = 1; i <= k; i++) {
        // 对当前层的每个POI
        for (int j : Vr[i-1]) {
            OS[i][j] = INF;
            parent[i][j] = -1;
            
            // 从前一层的所有状态转移
            for (auto& [l, cost_l] : OS[i-1]) {
                // 剪枝1：成本超过阈值
                if (cost_l >= costthresh) continue;
                
                // 剪枝2：当前成本+到终点的距离超过上界
                //if (i > 1 && cost_l + getExactDistance(l, vd) > wub) {
                //    continue;
                //}
                
                // 关键改进：直接计算精确距离（会自动缓存）
                double edgeCost = getExactDistance(l, j);
                double newCost = cost_l + edgeCost;
                
                // 剪枝3：新成本超过阈值
                if (newCost >= costthresh) continue;
                
                // 更新最优解
                if (newCost < OS[i][j]) {
                    OS[i][j] = newCost;
                    parent[i][j] = l;
                }
            }
        }
    }
    
    // 找到连接终点的最优路径
    minTotalCost = INF;
    int bestLastPOI = -1;
    
    for (auto& [lastPOI, costToLast] : OS[k]) {
        if (costToLast >= INF) continue;
        
        double distToEnd = getExactDistance(lastPOI, vd);
        double totalCost = costToLast + distToEnd;
        
        if (totalCost < minTotalCost) {
            minTotalCost = totalCost;
            bestLastPOI = lastPOI;
        }
    }
    
    // 如果没找到有效路径
    if (bestLastPOI == -1) {
        optimalPath.clear();
        minTotalCost = INF;
        return;
    }
    
    // 重构路径
    optimalPath.clear();
    int curr = bestLastPOI;
    for (int i = k; i >= 1; i--) {
        optimalPath.push_back(curr);
        curr = parent[i][curr];
    }
    
    reverse(optimalPath.begin(), optimalPath.end());
}

// 优化后的computeRouteDistance：纯查表，几乎零开销
double computeRouteDistance(int vs, const vector<int>& Rguide, int vd) {
    if (Rguide.empty()) return INF;
    
    double totalDistance = 0.0;
    
    // 辅助函数：快速查表（应该都已缓存）
    auto getDistance = [](int u, int v) -> double {
        if (u == v) return 0.0;
        pair<int,int> key = {min(u, v), max(u, v)};
        
        // 优先从缓存读取
        if (exactDistances.count(key)) {
            return exactDistances[key];
        }
        
        // 理论上不应该走到这里（OSE应该已经缓存了）
        // 但为了安全还是计算一下
        double dist = h2hQuery(u, v);
        exactDistances[key] = dist;
        return dist;
    };
    
    // vs -> Rguide[0]
    totalDistance += getDistance(vs, Rguide[0]);
    
    // Rguide[i-1] -> Rguide[i]
    for (size_t i = 1; i < Rguide.size(); i++) {
        totalDistance += getDistance(Rguide[i-1], Rguide[i]);
    }
    
    // Rguide[last] -> vd
    totalDistance += getDistance(Rguide.back(), vd);
    
    return totalDistance;
}

double greedySearch(int start, int end, vector<int>& greedyPath) {
    greedyPath.clear();
    double totalCost = 0;
    int current = start;
    
    for (int i = 0; i < Vr.size(); i++) {
        double minDist = INF;
        int bestPOI = -1;
        
        for (int poi : Vr[i]) {
            double dist = h2hQuery(current, poi);
            if (dist < minDist) {
                minDist = dist;
                bestPOI = poi;
            }
        }
        
        if (bestPOI == -1) return INF;
        
        greedyPath.push_back(bestPOI);
        totalCost += minDist;
        current = bestPOI;
    }
    
    totalCost += h2hQuery(current, end);
    return totalCost;
}

double ROSEQuery(int s, int t, vector<vector<double>> &queryEmbeddings, vector<vector<int>> &similarCategories, vector<int> &path) {
    // Algorithm 1: Line 1 - 初始化
    double costguide = 0.0;
    double costexact = 0.0;
    
    K = queryEmbeddings.size();
    path.clear();
    Vr.resize(K);
    exactDistances.clear(); // 每次查询清空精确距离缓存
    
    // 构建候选POI集合
    for (int i = 0; i < K; ++i) {
        set<int> uniqueNodesVr;        
        for (int catId : similarCategories[i]) {            
            for (int nodeId : category2Nodes[catId]) {                
                uniqueNodesVr.insert(nodeId);
            }
        }
        Vr[i] = vector<int>(uniqueNodesVr.begin(), uniqueNodesVr.end());
    }
    
    // Algorithm 1: Line 2 - 通过贪心搜索获得精确成本
    vector<int> greedyPath;
    costexact = greedySearch(s, t, greedyPath);
    double wub;
    if (costexact == INF) {
        return -1;
    }
    
    // Algorithm 1: Line 3 - 初始化当前最优成本阈值
    double costthresh = costexact;
    
    // Algorithm 1: Line 4-10 - 主循环：while costguide ≠ costexact
    int iterations = 0;
    const int maxIterations = 100;
    
    while (iterations < maxIterations) {
        iterations++;
        
        // Algorithm 1: Line 5 - OSE找指导路径
        vector<int> Rguide;
        costguide = 0;
        wub = costexact;
        OSE(s, t, costthresh, Rguide, costguide, wub);

        if (Rguide.empty() || costguide == INF) {
            break;
        }
        
        // Algorithm 1: Line 6 - 计算指导路径的精确距离
        costexact = computeRouteDistance(s, Rguide, t);
        
        // Algorithm 1: Line 7-9 - 更新阈值
        if (costexact < costthresh) {
            costthresh = costexact;
        }
        
        // Algorithm 1: Line 4条件检查 - 严格等值判断
        if (abs(costguide - costexact) < 1e-2) {
            //cout << "ROSE converged after " << iterations << " iterations" << endl;
            break;
        }
        
        //cout << "Iteration " << iterations << ": costguide=" << costguide << ", costexact=" << costexact << endl;
    }
    
    return costexact;
}

EmbeddingService* g_embeddingService = nullptr;
int initTransformer(){
    try {
        // 创建embedding服务
        std::cout << "Initializing embedding service..." << std::endl;
        g_embeddingService = new EmbeddingService();
        
        // 你的查询文本
        /*std::vector<std::string> queries = {
            "quiet coffee place",
            "romantic dinner restaurant", 
            "outdoor sports gym",
            "luxury shopping mall",
            "emergency medical clinic"
        };
        
        std::cout << "Getting embeddings for " << queries.size() << " texts..." << std::endl;
        
        // 获取embeddings
        auto embeddings = g_embeddingService->getEmbeddings(queries);
        
        std::cout << "Successfully got " << embeddings.size() << " embeddings" << std::endl;
        std::cout << "Embedding dimension: " << embeddings[0].size() << std::endl;
        
        // 验证归一化
        for (size_t i = 0; i < embeddings.size(); ++i) {
            float norm = 0.0f;
            for (float val : embeddings[i]) {
                norm += val * val;
            }
            norm = sqrt(norm);
            std::cout << "Query " << i << " (" << queries[i] << ") norm: " << norm << std::endl;
        }
        
        // 这里你可以继续你的buildIndex逻辑...
        std::cout << "\n Ready to build index with embeddings!" << std::endl;*/
        return 0; 
        
    } catch (const std::exception& e) {
        std::cerr << "Error in initTransformer: " << e.what() << std::endl;
        return -1;
    }
}

int main(int argc, char* argv[]) {
    string sfile, reqfile;
    int candidate_size, topk, catid, startqi;
    double alpha;
    if (argc > 1) {
        sfile = string(argv[1]);
    } else {
        sfile = string("NYC");
    }
    if (argc > 2) {
        reqfile = string(argv[2]);
        startqi = 4;
    } else {
        reqfile = string("req3");
        startqi = 0;
    }
    if (argc > 3) {
        alpha = stod(string(argv[3]));
    } else {
        alpha = 0.05;
    }
    cout << sfile << " " << reqfile << " " << alpha << endl;

    FILE *fp_query, *fp_network;
    string prefix = string("../data/") + sfile + string("/");
    FILE *fp_requirement = fopen((prefix + reqfile).c_str(), "r");
    char buf[256];
    vector<string> R;
    while (fgets(buf, sizeof(buf), fp_requirement)) {
        string line(buf);
        // 移除末尾的换行符
        if (!line.empty() && line[line.length()-1] == '\n') {
            line.erase(line.length()-1);
        }
        // 移除可能的回车符（Windows格式）
        if (!line.empty() && line[line.length()-1] == '\r') {
            line.erase(line.length()-1);
        }
        // 跳过空行
        if (!line.empty()) {
            R.push_back(line);
        }
    }
    fclose(fp_requirement);

    initTransformer();

    string graphfile = prefix + string("USA-road-d.") + sfile + (".gr");
    fp_network = fopen(graphfile.c_str(), "r");
    char ch, buffer[100];
    int u, v, intw;
    double w;
    char cat1;
    //read graph
    for (int i = 0; i < 4; i++)
        fgets(buffer, 90, fp_network);
    for (int i = 0; i < 4; i++)
        fgetc(fp_network);
    fscanf(fp_network, "%d%d", &N, &M);
    for (int i = 0; i < 3; i++)
        fgets(buffer, 90, fp_network);
    for (int i = 0; i < M; i++) {
        fscanf(fp_network, "%c%d%d%d", &ch, &u, &v, &intw);
        fgets(buffer, 90, fp_network);
        w = (double)intw / 10.0;
        u--;
        v--;
        //printf("%d %d %c%d %c\n", u, v, cat1, cat2, l);
        if (i % 2 == 0){
            G[u].push_back(ID(v, w));
            G[v].push_back(ID(u, w));
        }
    }

    double minx = DBL_MAX, miny = DBL_MAX, maxx = -DBL_MAX, maxy = -DBL_MAX;
    //coords
    string coordsfile = prefix + string("USA-road-d.") + sfile + (".co");
    FILE *fp_coords = fopen(coordsfile.c_str(), "r");
    for (int i = 0; i < 7; i++)
        fgets(buffer, 90, fp_coords);
    
    for (int i = 0; i < N; i++) {
        double x, y;
        fscanf(fp_coords, "%c%d%lf%lf", &ch, &u, &x, &y);
        fgets(buffer, 90, fp_coords);
        x /= 1000000.0;
        y /= 1000000.0;
        coords.push_back(DD(x, y));
        minx = min(x, minx), miny = min(y, miny), maxx = max(x, maxx), maxy = max(y, maxy);
    }
    printf("%f %f %f %f\n", minx, maxx, miny, maxy);


    cout << "Building H2H Index for dataset: " << sfile << endl;
    // call h2h.cpp
    buildH2HIndex(sfile);
    cout << "H2H Index building completed!" << endl;

    std::chrono::high_resolution_clock::time_point t1, t2;
	std::chrono::duration<double> time_span;
	double runT;

    readPOINodeMapping(sfile);
    readCategoryID(sfile);
    readCategoryEmbeddings(sfile);

    t1=std::chrono::high_resolution_clock::now();
    buildRNII(50);
    t2=std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
	runT= time_span.count();
    cout<<"buildRNII Time "<<runT<<endl;

    // 获取查询文本的embedding后
    //vector<string> R = {"beach", "desert shop", "Yoga Studio"};
    //vector<string> R = {"get some fast food", "buy a birthday gift", "fill up gas"};
    //vector<string> R = {"repair a tire", "buy some video games", "go to a flea market"};
    vector<vector<double>> queryEmb;
    g_embeddingService->getEmbeddings(R, queryEmb);
    K = queryEmb.size();
    vector<vector<int> > similarCategories(K);
    for (int i = 0; i < K; ++i) {
        findSimilarCategories(queryEmb[i], alpha, similarCategories[i]);
    }

    vector<int> path;
    t1=std::chrono::high_resolution_clock::now();
    int s = 0, t = 9;
    cout << ROSEQuery(s, t, queryEmb, similarCategories, path) << endl;
    t2=std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
	runT= time_span.count();
    cout<<"ROSE Query Time "<<runT<<endl;

    string setres;
    for (int qi = startqi; qi < 5; qi++){
        vector<II> queryset;
        vector<double> ans;
        string s3 = string("../data/") + sfile + string("/") + string("q") + to_string(qi + 1);
        fp_query = fopen(s3.c_str(), "r");
        int qs, qt;
        while (~fscanf(fp_query, "%d%d", &qs, &qt)){
            queryset.push_back(II(qs, qt));
        }
        fclose(fp_query);
        int qn = queryset.size();
        t1=std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; i++){
            int idx = i; // rand() % qn;
            ans.push_back(ROSEQuery(queryset[idx].first-1, queryset[idx].second-1, queryEmb, similarCategories, path));
        }
        t2=std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
        runT= time_span.count();
        cout << (string("q") + to_string(qi + 1)).c_str() << " Query Time\n" << runT*1000 << endl;
        setres += string("q") + to_string(qi + 1);
        setres += string(" Query Time \n") + to_string(runT*1000) + string("\n");
    }

    FILE *fp_record = fopen("ROSERecord.txt", "a");
    fprintf(fp_record, "%s\n", sfile.c_str());
    fprintf(fp_record, "%s\n", setres.c_str());
    fclose(fp_record);

    return 0;
}