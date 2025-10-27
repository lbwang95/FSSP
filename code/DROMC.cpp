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

// 查找相似度大于阈值的category编号
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

int t; // 全局终点，供heuristic()使用
unordered_map<int, double> poiToDestDist;  // 预计算每个POI到终点的距离
map<int, double> heuristicTable;
vector<vector<int>> Vr;
void precomputeHeuristics(int s) {
    //cout << "Precomputing heuristic table..." << endl;
    
    vector<set<int>> possibleNodesByCat(K);
    possibleNodesByCat[0].insert(s);  // catIdx=0时只有起点

    for (int catIdx = 1; catIdx < K; catIdx++) {
        // 继承前一个category的所有节点
        possibleNodesByCat[catIdx] = possibleNodesByCat[catIdx-1];
        // 加上前一个category的POI（因为访问完Vr[catIdx-1]后才到catIdx）
        for (int poi : Vr[catIdx-1]) {
            possibleNodesByCat[catIdx].insert(poi);
        }
    }

    // 然后计算启发式
    for (int catIdx = 0; catIdx < K; catIdx++) {
        for (int node : possibleNodesByCat[catIdx]) {
            double minCost = DBL_MAX;
            for (int poi : Vr[catIdx]) {
                double cost = h2hQuery(node, poi) + poiToDestDist[poi];
                minCost = min(minCost, cost);
            }
            heuristicTable[node*10+catIdx] = minCost;
        }
    }
}

struct State {
    int node;
    int categoryIdx;  // 已访问的类别数 (0到K)
    double cost;
    vector<int> visitedPOIs;
    
    double f() const {
        return cost + heuristic();
    }
    
    double heuristic() const {
        if (categoryIdx == K) return h2hQuery(node, t);

        auto it = heuristicTable.find(node*10+ categoryIdx);
        if (it != heuristicTable.end()) {
            return it->second;
        }
        return 0;
    }

};

struct StateComparator {
    bool operator()(const State& a, const State& b) const {
        return a.f() > b.f();  // 最小堆
    }
};

double DROMCQuery(int s, int t_param, vector<vector<double>> &queryEmbeddings, vector<vector<int>> &similarCategories, vector<int> &path) {
    t = t_param; // 设置全局终点
    K = queryEmbeddings.size();
    path.clear();
    Vr.resize(K);

    set<int> allPOIs;
    for (int i = 0; i < K; ++i) {
        set<int> uniqueNodesVr;        
        for (int catId : similarCategories[i]) {            
            for (int nodeId : category2Nodes[catId]) {                
                uniqueNodesVr.insert(nodeId);
                allPOIs.insert(nodeId);
            }
        }
        Vr[i] = vector<int>(uniqueNodesVr.begin(), uniqueNodesVr.end());
    }

    for (int poi : allPOIs) {
        poiToDestDist[poi] = h2hQuery(poi, t);
    }
    precomputeHeuristics(s);

    priority_queue<State, vector<State>, StateComparator> pq;
    map<pair<int, int>, double> visited; // (node, categoryIdx) -> minCost
    
    // 初始状态
    State start;
    start.node = s;
    start.categoryIdx = 0;
    start.cost = 0;
    start.visitedPOIs.clear();
    
    pq.push(start);
    
    while (!pq.empty()) {
        State current = pq.top();
        pq.pop();
        
        // 检查是否已访问过相同状态且代价更小
        pair<int, int> stateKey = {current.node, current.categoryIdx};
        if (visited.count(stateKey) && visited[stateKey] <= current.cost) {
            continue;
        }
        visited[stateKey] = current.cost;
        
        // 检查是否到达终点且访问完所有类别
        if (current.categoryIdx == K && current.node == t) {
            path = current.visitedPOIs;
            return current.cost;
        }
        
        // 如果还需要访问类别
        if (current.categoryIdx < K) {
            // 尝试访问下一个类别的所有POI
            for (int poi : Vr[current.categoryIdx]) {
                State next;
                next.node = poi;
                next.categoryIdx = current.categoryIdx + 1;
                next.cost = current.cost + h2hQuery(current.node, poi);
                next.visitedPOIs = current.visitedPOIs;
                next.visitedPOIs.push_back(poi);
                pq.push(next);
            }
        } else {
            // 已访问完所有类别，前往终点
            if (current.node != t) {
                State next;
                next.node = t;
                next.categoryIdx = K;
                next.cost = current.cost + h2hQuery(current.node, t);
                next.visitedPOIs = current.visitedPOIs;
                pq.push(next);
            }
        }
    }
    
    return -1; // 无解
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
    cout << DROMCQuery(s, t, queryEmb, similarCategories, path) << endl;
    t2=std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
	runT= time_span.count();
    cout<<"DROMC Query Time "<<runT<<endl;

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
            ans.push_back(DROMCQuery(queryset[idx].first-1, queryset[idx].second-1, queryEmb, similarCategories, path));
        }
        t2=std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
        runT= time_span.count();
        cout << (string("q") + to_string(qi + 1)).c_str() << " Query Time\n" << runT*1000 << endl;
        setres += string("q") + to_string(qi + 1);
        setres += string(" Query Time \n") + to_string(runT*1000) + string("\n");
    }

    FILE *fp_record = fopen("DROMCRecord.txt", "a");
    fprintf(fp_record, "%s\n", sfile.c_str());
    fprintf(fp_record, "%s\n", setres.c_str());
    fclose(fp_record);

    return 0;
}