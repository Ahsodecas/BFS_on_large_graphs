# BFS_on_large_graphs

Implementation of BFS algorithm for GPU with optimization for large graphs. 
* The first approach uses a frontier queue with vertices to be visited in the next iteration and atomic operations.
* The second approach uses multiple buffers from shared to global memory to decrease the number of atomic operations used.


The results for a graph with 75889 vertices and 508837 edges:
![image](https://github.com/user-attachments/assets/51c1a66e-ed4c-4071-8e12-9c53e938f707)


The results for a graph with 4847571 vertices and 68993773 edges:
![image](https://github.com/user-attachments/assets/afe78602-8239-4e8e-b47a-09e3a282aeba)

