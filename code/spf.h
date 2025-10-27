#ifndef SPF_H
#define SPF_H
#include "h2h.h"

extern vector<ID> G[MAX_V];
extern map<int, vector<int>> category2Nodes;

// 函数声明 - buildIndex.cpp 需要调用的函数
void buildAllSPFCategoryIndices(int n);
vector<int> applySPFPruning(int u, const vector<int> &Vr, vector<int> &similarCategories);

#endif // SPF_H