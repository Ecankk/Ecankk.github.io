---
作者: Ecank
tags:
  - 图论
是否完成: 
文档性质(子/父/无): 
想法: 
keyword: 
title: 最近公共祖先LCA
modify: 2025-03-26 19:10
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

## 倍增
```cpp
#include <bits/stdc++.h>
using namespace std;
#define endl '\n'
#define INF 0x3f3f3f3f3f3f3f3f
#define LL long long
const int MAXN = 1e5 + 5;
const double EPS = 1e-9;
int n, m, s, a, b;

vector<vector<int>> g;
vector<int> dep;
vector<vector<int>> fa;
void dfs(int u, int father)
{
    dep[u] = dep[father] + 1;
    fa[u][0] = father;
    for (int i = 1; i <= 22; i++) {
        fa[u][i] = fa[fa[u][i - 1]][i - 1];
    }
    for (auto ele : g[u]) {
        if (ele == father) continue;
        dfs(ele, u);
    }
}
int lca(int u, int v)
{
    if (dep[u] < dep[v]) swap(u, v);
    for (int i = 22; i >= 0; i--) {
        if (dep[fa[u][i]] >= dep[v]) u = fa[u][i];
    }
    if (u == v) return u;
    for (int i = 22; i >= 0; i--) {
        if (fa[u][i] != fa[v][i]) {
            u = fa[u][i];
            v = fa[v][i];
        }
    }
    return fa[u][0];
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cin >> n >> m >> s;
    g.assign(n + 1, vector<int>());
    dep.resize(n + 1);
    fa.assign(n + 1, vector<int>(23, 0));
    for (int i = 1; i < n; i++) {
        cin >> a >> b;
        g[a].push_back(b);
        g[b].push_back(a);
    }
    dfs(s, 0);
    while (m--) {
        cin >> a >> b;
        cout << lca(a, b) << endl;
    }
    return 0;
}
```