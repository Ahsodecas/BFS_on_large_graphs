#include "graph.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <queue>
#include <cstdio>
#include <chrono>

#define GRAPH1_FILENAME "C:\\Users\\nasty\\source\\repos\\bfs_on_large_graphs\\bfs_on_large_graphs\\x64\\Debug\\graph1.txt"
#define GRAPH2_FILENAME "C:\\Users\\nasty\\source\\repos\\bfs_on_large_graphs\\bfs_on_large_graphs\\x64\\Debug\\graph2.txt"
#define THREADS_PER_BLOCK 128

void check_error(cudaError_t error, std::string message) 
{
    if (error != cudaSuccess) 
    {
        std::cerr << message << " " << error;
        exit(1);
    }
}

void check_cpu_gpu_output(vector<int>& cpuDistance, vector<int>& gpuDistance, Graph& G) 
{
    for (int i = 0; i < G.numVertices; i++) 
    {
        if (gpuDistance[i] != cpuDistance[i]) 
        {
            std::cout << i << " " << gpuDistance[i] << " " << cpuDistance[i] << endl;
            std::cout << "Output does not match" << endl;
            exit(1);
        }
    }
    std::cout << "Correct output" << endl;
}

void print_path(vector<int>& distance, vector<int>& parents, Graph& G, int vertex) 
{
    if (distance[vertex] == INT_MAX)
    {
        std::cout << "No path exists" << endl;
        return;
    }
    int parent = parents[vertex];
    std::cout << vertex << " <--- " << parent;
    while (parent != parents[parent])
    {
        parent = parents[parent];
        std::cout << " <--- " << parent;
    }
    std::cout << endl;
}

__global__ void bfs_gpu_mark(int* edgesOffset, int* adjacencyList, int* edgesSize, int* distance, int* currQueue, int* nextQueueFlags, int currQueueSize, int* nextQueueSize, int* parent)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId < currQueueSize)
    {
        int vertex1 = currQueue[threadId];
        int startIdx = edgesOffset[vertex1];
        int endIdx = edgesOffset[vertex1] + edgesSize[vertex1];

        for (int i = startIdx; i < endIdx; i++)
        {
            int vertex2 = adjacencyList[i];

            if (atomicCAS(&distance[vertex2], INT_MAX, distance[vertex1] + 1) == INT_MAX)
            {
                nextQueueFlags[vertex2] = 1;
                parent[vertex2] = vertex1;
            }
        }
    }
}

__global__ void bfs_gpu_scan(int* nextQueueFlags, int* nextQueue, int numVertices, int* blockSums, int* nextQueueSize)
{
    __shared__ int temp[THREADS_PER_BLOCK];

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId >= numVertices) 
        return;

    temp[threadIdx.x] = nextQueueFlags[threadId];
    __syncthreads();

    // prefix sum canculation within the blocks
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int val = 0;
        if (threadIdx.x >= stride)
        {
            val = temp[threadIdx.x - stride];
        }
        __syncthreads();
        temp[threadIdx.x] += val;
        __syncthreads();
    }
    __syncthreads();

    // save block sums and increment the next queue size
    if (threadIdx.x == blockDim.x - 1 || (numVertices % THREADS_PER_BLOCK != 0 && blockIdx.x == (numVertices / THREADS_PER_BLOCK) && threadIdx.x == (numVertices % THREADS_PER_BLOCK - 1)))
    {
        blockSums[blockIdx.x] = temp[threadIdx.x];  
        atomicAdd(nextQueueSize, blockSums[blockIdx.x]);
    }
}

__global__ void blockwise_gpu_scan(int* blockSums, int blocksNum, int* nextQueueSize)
{
    for (int i = 1; i < blocksNum; i++)
    {
        blockSums[i] += blockSums[i - 1];
    }
}

__global__ void bfs_gpu_form_next_queue(int* nextQueueFlags, int* nextQueue, int* nextQueueSize, int numVertices, int* blockSums, int blocksNum)
{
    __shared__ int temp[THREADS_PER_BLOCK];

    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId >= numVertices)
    {
        return;
    }

    temp[threadIdx.x] = nextQueueFlags[threadId];
    __syncthreads();

    // calculate the prefix sum
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int val = 0;
        if (threadIdx.x >= stride)
        {
            val = temp[threadIdx.x - stride];
        }
        __syncthreads();
        temp[threadIdx.x] += val;
        __syncthreads();
    }

    // add block sum 
    if (blockIdx.x > 0)
    {
        temp[threadIdx.x] += blockSums[blockIdx.x - 1];
    }
    __syncthreads();

    // update the next queue
    if (nextQueueFlags[threadId] == 1 && temp[threadIdx.x] > 0)
    {
        nextQueue[temp[threadIdx.x] - 1] = threadId;
    }
}

void launch_bfs_gpu_multibuffer(int startVertex, int** adjacencyList, int** edgesOffset, int** edgesSize, int** distance, int** parent, int* nextQueueSize, int** currQueue, int** nextQueue, int numVertices, int numEdges)
{
    auto startTime = std::chrono::steady_clock::now();
    int blocksNum = (numVertices + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    int* nextQueueFlags, *blockSums;
    check_error(cudaMalloc(&nextQueueFlags, numVertices * sizeof(int)), "cudaMalloc failed!");
    check_error(cudaMalloc(&blockSums, blocksNum * sizeof(int)), "cudaMalloc failed!");

    int currQueueSize = 1;
    int nextQueueSize_cpu = 0;
    check_error(cudaMemset(nextQueueSize, 0, sizeof(int)), "cudaMemset failed");

    while (currQueueSize)
    {
        check_error(cudaMemset(nextQueueFlags, 0, numVertices * sizeof(int)), "cudaMemset failed");
        check_error(cudaMemset(blockSums, 0, blocksNum * sizeof(int)), "cudaMemset failed");

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Kernel failed after memset of flags and blocks: %s\n", cudaGetErrorString(err));
            return;
        }

        bfs_gpu_mark << <currQueueSize / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> >(*edgesOffset, *adjacencyList, *edgesSize, *distance, *currQueue, nextQueueFlags, currQueueSize, nextQueueSize, *parent);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Kernel failed after mark: %s\n", cudaGetErrorString(err));
            return;
        }

        check_error(cudaMemset(nextQueueSize, 0, sizeof(int)), "cudaMemset failed");

        bfs_gpu_scan << <blocksNum, THREADS_PER_BLOCK >> > (nextQueueFlags, *nextQueue, numVertices, blockSums, nextQueueSize);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Kernel failed after scan : %s\n", cudaGetErrorString(err));
            return;
        }

        blockwise_gpu_scan << <1, 1 >> > (blockSums, blocksNum, nextQueueSize);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Kernel failed after scan kernel: %s\n", cudaGetErrorString(err));
            return;
        }
        
        bfs_gpu_form_next_queue << <blocksNum, THREADS_PER_BLOCK >> >(nextQueueFlags, *nextQueue, nextQueueSize, numVertices, blockSums, blocksNum);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Kernel failed after form next queue: %s\n", cudaGetErrorString(err));
            return;
        }

        check_error(cudaMemcpy(&currQueueSize, nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed in bfs multibuffer currQueueSize!");

        swap(*currQueue, *nextQueue);

        check_error(cudaMemcpy(nextQueueSize, &nextQueueSize_cpu, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed in bfs multibuffer nextQueueSize!");
    }

    check_error(cudaFree(nextQueueFlags), "cudaFree failed!");
    check_error(cudaFree(blockSums), "cudaFree failed!");

    auto endTime = std::chrono::steady_clock::now();
    std::cout << "Elapsed time for bfs multibuffer in milliseconds : " << (endTime - startTime).count() << " ms" << std::endl;
}

__global__ void bfs_gpu_queue(int* adjacencyList, int* edgesOffset, int* edgesSize, int currQueueSize, int* nextQueueSize, int* currQueue, int* nextQueue, int level, int* distance, int* parent)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < currQueueSize)
    {
        int vertex1 = currQueue[threadId];
        int startIdx = edgesOffset[vertex1];
        int endIdx = edgesOffset[vertex1] + edgesSize[vertex1];

        for (int i = startIdx; i < endIdx; i++)
        {
            int vertex2 = adjacencyList[i];
            if (atomicCAS(&distance[vertex2], INT_MAX, level + 1) == INT_MAX)
            {
                parent[vertex2] = vertex1;
                int queuePosition = atomicAdd(nextQueueSize, 1);
                nextQueue[queuePosition] = vertex2;
            }
        }
    }
}

void launch_bfs_gpu_queue(int startVertex, int** adjacencyList, int** edgesOffset, int** edgesSize, int** distance, int** parent, int* nextQueueSize, int** currQueue, int** nextQueue)
{
    auto startTime = std::chrono::steady_clock::now();

    int currQueueSize = 1;
    int nextQueueSize_cpu = 0;

    int level = 0;
    while (currQueueSize)
    {
        bfs_gpu_queue << <currQueueSize / THREADS_PER_BLOCK + 1, THREADS_PER_BLOCK >> > (*adjacencyList, *edgesOffset, *edgesSize, currQueueSize, nextQueueSize, *currQueue, *nextQueue, level, *distance, *parent);
        
        cudaDeviceSynchronize();

        check_error(cudaMemcpy(&currQueueSize, nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed!");

        swap(*currQueue, *nextQueue);
        level++;

        check_error(cudaMemcpy(nextQueueSize, &nextQueueSize_cpu, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed!");
    }

    auto endTime = std::chrono::steady_clock::now();
    std::cout << "Elapsed time for bfs queue in milliseconds : " << (endTime - startTime).count() << " ms" << std::endl;
}

void bfs_cpu(Graph& graph, int start, vector<int>& distance, vector<int>& parent)
{
    auto startTime = std::chrono::steady_clock::now();
    queue<int> q;
    vector<bool> visited(graph.numVertices, false);

    q.push(start);
    visited[start] = true;
    distance[start] = 0;
    parent[start] = start;

    while (!q.empty()) 
    {
        int size = q.size();
        for (int i = 0; i < size; i++)
        {
            int vertex1 = q.front();
            q.pop();
            int startIdx = graph.edgesOffset[vertex1];
            int endIdx = graph.edgesOffset[vertex1] + graph.edgesSize[vertex1];

            for (int i = startIdx; i < endIdx; i++)
            {
                int vertex2 = graph.adjacencyList[i];
                if (!visited[vertex2])
                {
                    visited[vertex2] = true;
                    distance[vertex2] = distance[vertex1] + 1;
                    parent[vertex2] = vertex1;
                    q.push(vertex2);
                }
            }
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    std::cout << "Elapsed time for bfs cpu in milliseconds : " << (endTime - startTime).count() << " ms" << std::endl;
}


cudaError_t init_gpu(Graph& g, int startingVertex, int** adjacencyList, int** edgesOffset, int** edgesSize, int** distance, int** parent, int** nextQueueSize, int** currQueue, int** nextQueue, vector<int>& distance_gpu, vector<int>& distance_cpu, vector<int>& parent_cpu)
{
    check_error(cudaMalloc(adjacencyList, g.numEdges * sizeof(int)), "cudaMalloc failed!");
    check_error(cudaMalloc(edgesOffset, g.numVertices * sizeof(int)), "cudaMalloc failed!");
    check_error(cudaMalloc(edgesSize, g.numVertices * sizeof(int)), "cudaMalloc failed!");
    check_error(cudaMalloc(distance, g.numVertices * sizeof(int)), "cudaMalloc failed!");
    check_error(cudaMalloc(parent, g.numVertices * sizeof(int)), "cudaMalloc failed!");
    check_error(cudaMalloc(currQueue, g.numVertices * sizeof(int)), "cudaMalloc failed!");
    check_error(cudaMalloc(nextQueue, g.numVertices * sizeof(int)), "cudaMalloc failed!");
    check_error(cudaMalloc(nextQueueSize, sizeof(int)), "cudaMalloc failed!");

    check_error(cudaMemcpy(*adjacencyList, g.adjacencyList.data(), g.numEdges * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed!");
    check_error(cudaMemcpy(*edgesOffset, g.edgesOffset.data(), g.numVertices * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed!");
    check_error(cudaMemcpy(*edgesSize, g.edgesSize.data(), g.numVertices * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed!");
    check_error(cudaMemcpy(*distance, distance_cpu.data(), g.numVertices * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed!");
    check_error(cudaMemcpy(*parent, parent_cpu.data(), g.numVertices * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed!");
    check_error(cudaMemcpy(*currQueue, &startingVertex, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed!");

    return cudaSuccess;
}

cudaError_t cleanup_gpu(int* adjacencyList, int* edgesOffset, int* edgesSize, int* distance, int* parent, int* nextQueueSize, int* currQueue, int* nextQueue)
{
    check_error(cudaFree(adjacencyList), "cudaFree failed!");
    check_error(cudaFree(edgesOffset), "cudaFree failed!");
    check_error(cudaFree(edgesSize), "cudaFree failed!");
    check_error(cudaFree(distance), "cudaFree failed!");
    check_error(cudaFree(parent), "cudaFree failed!");
    check_error(cudaFree(currQueue), "cudaFree failed!");
    check_error(cudaFree(nextQueue), "cudaFree failed!");
    check_error(cudaFree(nextQueueSize), "cudaFree failed!");

    return cudaSuccess;
}

cudaError_t reset_gpu(Graph& g, int* startingVertex, int** distance, int** parent, int* nextQueueSize, int** currQueue, int** nextQueue)
{
    vector<int> distance_cpu(g.numVertices, INT_MAX);
    vector<int> parent_cpu(g.numVertices, INT_MAX);
    distance_cpu[*startingVertex] = 0;
    parent_cpu[*startingVertex] = *startingVertex;

    check_error(cudaMemcpy(*distance, distance_cpu.data(), g.numVertices * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed!");
    check_error(cudaMemcpy(*parent, parent_cpu.data(), g.numVertices * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed!");
    check_error(cudaMemset(*currQueue, 0, g.numVertices * sizeof(int)), "cudaMemset failed!");
    check_error(cudaMemcpy(*currQueue, startingVertex, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy failed!");
    check_error(cudaMemset(*nextQueue, 0, g.numVertices * sizeof(int)), "nextQueue failed!");
    check_error(cudaMemset(nextQueueSize, 0, sizeof(int)), "cudaMemset failed!");

    return cudaSuccess;
}

int main()
{
    Graph g(GRAPH1_FILENAME);
    // Graph g(GRAPH2_FILENAME);

    int startingVertex = 0;

    int* adjacencyList, *edgesOffset, *edgesSize, *distance, *parent, *currQueue, *nextQueue;
    int* nextQueueSize;
    vector<int> distance_cpu(g.numVertices, INT_MAX);
    vector<int> parent_cpu(g.numVertices, INT_MAX);
    distance_cpu[startingVertex] = 0;
    parent_cpu[startingVertex] = startingVertex;
    vector<int> distance_gpu(g.numVertices, INT_MAX);
    vector<int> parent_gpu(g.numVertices, INT_MAX);

    check_error(init_gpu(g, startingVertex, &adjacencyList, &edgesOffset, &edgesSize, &distance, &parent, &nextQueueSize, &currQueue, &nextQueue, distance_gpu, distance_cpu, parent_cpu), "init_gpu failed");

    bfs_cpu(g, startingVertex, distance_cpu, parent_cpu);

    launch_bfs_gpu_queue(startingVertex, &adjacencyList, &edgesOffset, &edgesSize, &distance, &parent, nextQueueSize, &currQueue, &nextQueue);
    
    check_error(cudaMemcpy(distance_gpu.data(), distance, g.numVertices * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed!");
    check_error(cudaMemcpy(parent_gpu.data(), parent, g.numVertices * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed!");
    check_cpu_gpu_output(distance_cpu, distance_gpu, g);

    check_error(reset_gpu(g, &startingVertex, &distance, &parent, nextQueueSize, &currQueue, &nextQueue), "reset_gpu failed");

    launch_bfs_gpu_multibuffer(startingVertex, &adjacencyList, &edgesOffset, &edgesSize, &distance, &parent, nextQueueSize, &currQueue, &nextQueue, g.numVertices, g.numEdges);
    
    check_error(cudaMemcpy(distance_gpu.data(), distance, g.numVertices * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed!");
    check_error(cudaMemcpy(parent_gpu.data(), parent, g.numVertices * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy failed!");
    check_cpu_gpu_output(distance_cpu, distance_gpu, g);

    /*
    int target_vertex = 23333;
    printPath(distance_cpu, parent_cpu, g, target_vertex);
    printPath(distance_gpu, parent_gpu, g, target_vertex);*/

    check_error(cleanup_gpu(adjacencyList, edgesOffset, edgesSize, distance, parent, nextQueueSize, currQueue, nextQueue), "cleanup_gpu failed");

    check_error(cudaDeviceReset(), "cudaDeviceReset failed!");

    return 0;
}
