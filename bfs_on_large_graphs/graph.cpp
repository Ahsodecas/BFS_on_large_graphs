#include "graph.hpp"
#include <fstream>

void Graph::buildGraph(string fileName)
{
	// read the data from the dataset
	ifstream dataFile(fileName);
	if (dataFile.is_open())
	{
		cout << "The file is being read" << endl;
		int vertex1, vertex2;
		dataFile >> this->numVertices >> this->numEdges;
		this->numVertices += 4;
		int approximateMaxDegree = this->numEdges / this->numVertices;

		vector<int> v;
		v.reserve(approximateMaxDegree);
		vector<vector<int>> adjacencyList_temp(numVertices, v);

		while (dataFile >> vertex1 >> vertex2)
		{
			adjacencyList_temp[vertex1].push_back(vertex2);
		}
		dataFile.close();

		// convert the data to acceptable format
		for (int vertex1 = 0; vertex1 < this->numVertices; vertex1++)
		{
			this->edgesOffset.push_back(this->adjacencyList.size());
			this->edgesSize.push_back(adjacencyList_temp[vertex1].size());
			for (int vertex2 : adjacencyList_temp[vertex1])
			{
				this->adjacencyList.push_back(vertex2);
			}
		}
	}
	else
	{
		cout << "The file is not found" << endl;
	}
}
