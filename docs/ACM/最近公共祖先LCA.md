---
作者: Ecank
tags:
  - "#树"
是否完成: 
文档性质(子/父/无): 
想法: 
keyword: 
title: 最近公共祖先LCA
modify: 2025-03-17 15:34
category: docs/ACM
share: true
---
## 树链剖分
使用两边 dfs 进行树链剖分，让两个节点跳转后位于同一链条，层次低的就是最近公共祖先
```c++
#include <bits/stdc++.h>
using namespace std;
#define endl '\n'
#define INF 0x3f3f3f3f
#define LL long long
const int MAXN = 5e5 + 5;
const double EPS = 1e-9;
vector<int> G[MAXN];
int fa[MAXN], dep[MAXN], son[MAXN], sz[MAXN];
int top[MAXN];
void dfs1(int u, int father)
{
    fa[u] = father;
    dep[u] = dep[father] + 1;
    sz[u] = 1;
    for (int v : G[u]) {
        if (v == father) continue;
        dfs1(v, u);
        sz[u] += sz[v];
        if (sz[v] > sz[son[u]]) {
            son[u] = v;
        }
    }
}
void dfs2(int u, int t)
{
    top[u] = t;
    if (!son[u]) return;
    dfs2(son[u], t);
    for (int v : G[u]) {
        if (v == fa[u] || v == son[u]) continue;
        dfs2(v, v);
    }
}
int lca(int u, int v)
{
    while (top[u] != top[v]) {
        if (dep[top[u]] < dep[top[v]]) {
            swap(u, v);
        }
        u = fa[top[u]];
    }
    return dep[u] < dep[v] ? u : v;
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    int n, m, s;
    cin >> n >> m >> s;
    for (int i = 1; i < n; i++) {
        int x, y;
        cin >> x >> y;
        G[x].push_back(y);
        G[y].push_back(x);
    }
    int a, b;
    dfs1(s, s);
    dfs2(s, s);
    while (m--) {
        cin >> a >> b;
        cout << lca(a, b) << endl;
    }
    return 0;
}
```