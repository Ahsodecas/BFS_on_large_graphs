#include <iostream>
#include <vector>
using namespace std;

class Graph {

public:
    Graph(string fileName)
    {
        buildGraph(fileName);
    }
    vector<int> adjacencyList; 
    vector<int> edgesOffset;
    vector<int> edgesSize;

    int numVertices = 0;
    int numEdges = 0;

private:
    void buildGraph(string fileName);
};








