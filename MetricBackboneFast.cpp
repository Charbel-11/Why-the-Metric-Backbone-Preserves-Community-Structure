#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct pair_hash {
    inline std::size_t operator()(const std::pair<int,int> & v) const {
        return v.first*31+v.second;
    }
};

struct edge {
	int u, v; double w; edge() {}
	edge(int _u, int _v, double _w) :
		u(_u), v(_v), w(_w) {}
};
struct node { vector<edge> edges; };

struct graph {
    int n, m;
    string path;
	vector<node> nodes;

    unordered_map<pair<int, int>, double, pair_hash> edgesMB;
	graph(const string& _path) : path(_path) {
        ifstream ifs("../" + path);
        ifs >> n >> m;
        nodes.resize(n);
        for(int i = 0; i < m; i++){
            int u, v; double w; ifs >> u >> v >> w;
            add_edge(u, v, w);
        }
        ifs.close();
    }

	void add_edge(int u, int v, double w) {
		nodes[u].edges.emplace_back(u, v, w);
		nodes[v].edges.emplace_back(v, u, w);
	}

    void getMB(bool approx=false){
        vector<int> nodesToTry;
        for(int i = 0; i < n; i++){ nodesToTry.push_back(i); }
        if (approx){
            int cnt = 2*log(n) + 1;
            auto rd = std::random_device {};
            auto rng = std::default_random_engine { rd() };
            shuffle(nodesToTry.begin(), nodesToTry.end(), rng);
            nodesToTry.erase(nodesToTry.begin() + cnt, nodesToTry.end());
        }

        for (auto s : nodesToTry){
            vector<pair<int, double>> parent(n, {-1,-1});
            getSPT(s, parent);
            for(int u = 0; u < n; u++){
                auto [v, w] = parent[u];
                if (v == -1) { continue; }
                edgesMB[{u, v}] = w;
            }
        }
    }

    void getSPT(int s, vector<pair<int, double>>& parent){
        vector<double> dist(n, 1e15);
        vector<bool> visited(n, false);
        priority_queue <pair<ll, int>, vector<pair<ll, int>>, greater<>> pq;
        dist[s] = 0ll; pq.emplace( 0, s );

        while(!pq.empty()) {
            int cur = pq.top().second; pq.pop();
            if (visited[cur]) { continue; }
            visited[cur] = true;

            for (auto &e : nodes[cur].edges)
                if (dist[e.v] > dist[cur] + e.w) {
                    dist[e.v] = dist[cur] + e.w;
                    parent[e.v] = {e.u, e.w};
                    pq.emplace( dist[e.v], e.v );
                }
        }
    }

    void writeMBToFile(bool approx = false){
        string curPath = "../MB_" + path;
        if (approx){ curPath = "../MBApprox_" + path; }
        ofstream ofs(curPath);
        ofs << n << '\n';
        for (auto& [p, w] : edgesMB){
            ofs << p.first << " " << p.second << " " << w << '\n';
        }
        ofs.close();
    }

    void check(){
        vector<vector<double>> dist2(n, vector<double>(n, 1e8));
        for(int i = 0; i < n; i++){
            dist2[i][i] = 0;
            for(auto &e : nodes[i].edges){
                dist2[e.u][e.v] = e.w;
            }
        }
        for(int k = 0; k < n; k++){
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){
                    dist2[i][j] = min(dist2[i][j], dist2[i][k] + dist2[j][k]);
                }
            }
        }

        for(int i = 0; i < n; i++) {
            for(auto &e : nodes[i].edges){
                if (e.w == dist2[e.u][e.v]){
                    assert(edgesMB.count({e.u, e.v}));
                }
                else{
                    assert(!edgesMB.contains({e.u, e.v}));
                }
            }
        }
    }
};

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr), cout.tie(nullptr);

    bool approx = true;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    graph G("Cifar_122.txt");
    G.getMB(approx);
    G.writeMBToFile(approx);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time taken to build MB: " << std::chrono::duration_cast<std::chrono::seconds >(end - begin).count() << " s" << std::endl;

}