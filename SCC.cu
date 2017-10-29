#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
//#define DEBUG

#ifdef DEBUG
#define print(...) printf(__VA_ARGS__)
#else
#define print(...) ;
#endif

#define outs(...) print(__VA_ARGS__)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
  # error printf is only supported on devices of compute capability 2.0 and higher, please compile with -arch=sm_20 or higher    
#endif
int numVertices, numEdges;
int *Graph[2]; 

void CalculateBWSets(bool *visited, int *color, int vertex);
void CalculateFWSets(bool *visited, int *color, int vertex);

__device__ bool d_terminate;
__global__ void FW_Kernel(bool *visited, int *color, int *d_vertexArray, int *d_edgeArray, int numVertices) {

    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    int currentIndex, i;
    if (vertex == 0 || vertex > numVertices) return;
    currentIndex = d_vertexArray[vertex];
//    printf("Thread: %d. Terminate on thread: %d is %d.\n", vertex, vertex, d_terminate);
//    printf("Thread: %d. Vertex + 1 index = %d\n", vertex, d_vertexArray[vertex+1]);
    if (visited[vertex] == true) {
        for(i = d_edgeArray[currentIndex]; currentIndex < d_vertexArray[vertex+1];) {
//            printf("Thread: %d. Edge %d, %d\n", vertex, vertex, i);
            if (color[vertex] == color[i] && visited[i] == false) {
                visited[i] = true;
                d_terminate = false;
//                printf("Terminate on thread: %d is %d.\n", vertex, d_terminate);
            }
                currentIndex++;
                i = d_edgeArray[currentIndex];
        }
    }
}
__global__ void BW_Kernel(bool *visited, int *color, int *d_vertexArray, int *d_edgeArray, int numVertices) {

    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    int currentIndex, i;
    if (vertex == 0 || vertex > numVertices) return;
    currentIndex = d_vertexArray[vertex];
//    printf("Thread: %d, visited = %d\n", vertex, visited[vertex]);
    if (visited[vertex] == false) {
        for(i = d_edgeArray[currentIndex]; currentIndex < d_vertexArray[vertex+1];) {
//            printf("Thread: %d. Edge %d, %d. Color: %d, %d\n", vertex, vertex, i, color[vertex], color[i]);
            if (color[vertex] == color[i] && visited[i] == true) {
                visited[vertex] = true;
                d_terminate = false;
                print("Thread: %d. Added visited[%d] as true from node: %d.\n", vertex, vertex, i);
                break;
            }
                currentIndex++;
                i = d_edgeArray[currentIndex];
        }
    }
}
__device__ void dfsVisit(int root, bool *forward, bool *BWReach, int pitch2, int vertex, int *color, int *d_vertexArray, int *d_edgeArray) {

    int currentIndex = d_vertexArray[vertex];
    for(int i = d_edgeArray[currentIndex]; currentIndex < d_vertexArray[vertex+1];) {
        if (color[vertex] == color[i] && forward[i] == false) {
            outs("DFSVisit: Thread: %d Adding %d to forward of vertex: %d\n", root, i, root);
            forward[i] = true;
            bool *backward = (bool *)((char *)BWReach + i * pitch2);
            backward[root] = true;
            dfsVisit(root, forward, BWReach, pitch2, i, color, d_vertexArray, d_edgeArray);
        }
        currentIndex++;
        i = d_edgeArray[currentIndex];
    }
}
__global__ void CalculateAllFWBWSets(bool *FWReach, bool *BWReach, int pitch1, int pitch2, int *color, int *d_vertexArray, int *d_edgeArray, int *Queue, int pitch3, int numVertices) {

    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    outs("Calculate:Start thread: %d\n", vertex);
    if (vertex == 0 || vertex > numVertices)
        return;
    bool *forward = (bool *)((char *)FWReach + vertex * pitch1);
    bool *backward = (bool *)((char *)BWReach + vertex * pitch2);
    int *vertexQueue = (int *)((char *)Queue + vertex * pitch3);
    forward[vertex] = true;
    backward[vertex] = true;
    int currentIndex = d_vertexArray[vertex], head = 0;
    
    for(int i = d_edgeArray[currentIndex]; currentIndex < d_vertexArray[vertex+1];) {
        if (color[vertex] == color[i] && forward[i] == false) {
            outs("Calculate: Thread: %d Adding %d to forward of vertex: %d\n", vertex, i, vertex);
            forward[i] = true;
            backward = (bool *)((char *)BWReach + i * pitch2);
            backward[vertex] = true;

            //dfsVisit(vertex, forward, BWReach, pitch2, i, color, d_vertexArray, d_edgeArray);
            head++;
            vertexQueue[head] = i;
        }
        currentIndex++;
        i = d_edgeArray[currentIndex];
    }
    while (head != 0) {
        
        int currentVertex = vertexQueue[head];
        head--;
        currentIndex = d_vertexArray[currentVertex];
        for(int i = d_edgeArray[currentIndex]; currentIndex < d_vertexArray[currentVertex+1];) {
            if (color[vertex] == color[i] && forward[i] == false) {
                outs("Calculate: Thread: %d Adding %d to forward of vertex: %d\n", vertex, i, vertex);
                forward[i] = true;
                backward = (bool *)((char *)BWReach + i * pitch2);
                backward[vertex] = true;
    
                //dfsVisit(vertex, forward, BWReach, pitch2, i, color, d_vertexArray, d_edgeArray);
                head++;
                vertexQueue[head] = i;
            }
            currentIndex++;
            i = d_edgeArray[currentIndex];
        }
    }
    /*for(int i = d_edgeArray[currentIndex]; currentIndex < d_vertexArray[vertex+1];) {
        if (color[vertex] == color[i] && forward[i] == false) {
            outs("Calculate: Thread: %d Adding %d to forward of vertex: %d\n", vertex, i, vertex);
            forward[i] = true;
            *backward = (bool *)((char *)BWReach + i * pitch2);
            backward[vertex] = true;

            dfsVisit(vertex, forward, BWReach, pitch2, i, color, d_vertexArray, d_edgeArray);
        }
        currentIndex++;
        i = d_edgeArray[currentIndex];
    }*/
}
__global__ void InitReachVector(bool *Reach, int pitch, int n)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;

    bool* row = (bool*)((char*)Reach + r * pitch);
    for (int c = 0; c <= n; ++c) {
        row[c] = false;
    }
}
__global__ void CheckReachVector(bool *Reach, int pitch, int n)
{
    int r = threadIdx.x;

    bool* row = (bool*)((char*)Reach + r * pitch);
    print("Thread: %d : ", r);
    for (int c = 0; c <= n; ++c) {
        print("(%d -> :%d = %d), ", r, c, row[c]);
    }
}
__device__ int d_queueHead, d_queueTail;
__device__ int d_SubGraphColor;
__global__ void FWBWAlgo_Kernel(int *d_SCC, int *d_color, bool *d_mark, int *d_vertexArray, int *d_edgeArray, int *d_workQueue, int *d_workQueue2, int numVertices,/* int *d_SubGraphColor,*/ bool *FWReach, bool *BWReach, int pitch1, int pitch2, /*int *d_queueHead, int *d_queueTail,*/ int currentQueueHead, int currentQueueTail) {

    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int pivot = -1, i;
    if (thread >= currentQueueHead)
        return;
    if (d_workQueue[thread] == -1) {
        return;
    }
    int currentColor = d_workQueue[thread];
    for (i = 1; i <= numVertices; i++) {
        if (d_color[i] == currentColor && d_mark[i] == false) {
            pivot = i;
            break;
        }
    }
    outs("Start: Thread : %d. Color = %d\n", thread, currentColor);
    if (pivot == -1) {
        outs("Could not find an appropriate Pivot for color: %d\n", currentColor);
        d_workQueue[thread] = -1;
//        int temp2 = atomicAdd(&d_queueTail, 1);
        return;
    }
    d_terminate = false;
    outs("Thread: %d: Pivot is : %d\n", currentColor, pivot);
    int colorFW, colorBW, colorSCC;
    colorFW = atomicAdd(&d_SubGraphColor, 1);
    colorBW = atomicAdd(&d_SubGraphColor, 1);
    colorSCC = atomicAdd(&d_SubGraphColor, 1);

    outs("Thread: %d: Color FW: %d\n", currentColor, colorFW);
    outs("Thread: %d: Color BW: %d\n", currentColor, colorBW);
    outs("Thread: %d: Color SCC: %d\n", currentColor, colorSCC);
    
    bool *forward = (bool *)((char *)FWReach + pivot * pitch1);
    for (i = 1; i <= numVertices; i++) {
        if (forward[i] == true) {
            outs("Forward Reachable (%d, %d)\n", pivot, i);
            if (currentColor == d_color[i] && d_mark[i] == false) {
                outs("\tAdding to Forward Set (%d, %d)\n", pivot, i);
                d_color[i] = colorFW;
            }
        }
    }
    bool *backward = (bool *)((char *)BWReach + pivot * pitch2);
    for (i = 1; i <= numVertices; i++) {
        if (backward[i] == true) {
            outs("Backward Reachable (%d, %d)\n", pivot, i);
            if (currentColor == d_color[i] && d_mark[i] == false) {
                outs("Adding to Backward Set (%d, %d)\n", pivot, i);
                d_color[i] = colorBW;
            }else if (colorFW == d_color[i] && d_mark[i] == false) {
                d_color[i] = colorSCC;
                d_SCC[i] = colorSCC;
                d_mark[i] = true;
                outs("Adding %d to SCC of color %d\n", i, colorSCC);
            }
        }
    }
/*    d_workQueue[colorFW] = colorFW;
    d_workQueue[colorBW] = colorBW;
//    d_workQueue[colorSCC] = colorSCC;
*/    d_workQueue[thread] = -1;
//    int temp2 = atomicAdd(&d_queueTail, 1);
    
    int temp = atomicAdd(&d_queueTail, 1);
    d_workQueue2[temp] = colorFW;
    temp = atomicAdd(&d_queueTail, 1);
    d_workQueue2[temp] = colorBW;
    temp = atomicAdd(&d_queueTail, 1);
    //d_workQueue[temp] = colorSCC;
    d_workQueue2[temp] = currentColor;
//    outs("Thread Queue: Head: %d, Tail: %d\n", d_queueHead, d_queueTail);
}
void calculateSCC(int *SCC) {
    
    int temp = 1;
    cudaMemcpyToSymbol(d_SubGraphColor, &temp, sizeof(int), 0, cudaMemcpyHostToDevice);
    bool *mark = (bool *)malloc((numVertices + 1) * sizeof(bool));
    int *workQueue = (int *)malloc((1000) * sizeof(int));
    int *color = (int *)malloc((numVertices + 1) * sizeof(int));
    for (int i = 0; i <= numVertices; i++) {
        color[i] = 0;
        mark[i] = false;
    }

    for (int i = 0; i <= 1000; i++) {
        workQueue[i] = -1;
    }
    workQueue[0] = 0;
    int *d_vertexArray, *d_edgeArray, *d_color, *d_workQueue, *d_workQueue2, *d_SCC;
    int size1 = (numVertices + 2) * sizeof (int);
    int size2 = (numEdges + 1) * sizeof (int);
    bool *FWReach, *BWReach, *d_mark;
    size_t pitch1, pitch2;
    int numThreadsPerBlock = 256;
    int numBlocksPerGrid = (numVertices + 1 + numThreadsPerBlock - 1) / numThreadsPerBlock;
    cudaMallocPitch((void**)&FWReach, &pitch1, (numVertices + 1) * sizeof(bool), (numVertices + 1));
    cudaMallocPitch((void**)&BWReach, &pitch2, (numVertices + 1) * sizeof(bool), (numVertices + 1));
    InitReachVector<<<numBlocksPerGrid, numThreadsPerBlock>>>(FWReach, pitch1, (numVertices + 1));
    InitReachVector<<<numBlocksPerGrid, numThreadsPerBlock>>>(BWReach, pitch2, (numVertices + 1));
    
    int size, sizeBool = (numVertices + 1) * sizeof(bool);
    bool *visited = (bool *)malloc(sizeBool);
    size = (numVertices + 1) * sizeof(int);
    int  i;
    for(i = 0; i <= numVertices; i++) {
        visited[i] = false;
        color[i] = 0;
    }
    cudaError_t err;


    err=cudaMalloc((void **)&d_vertexArray, (numVertices + 2) * sizeof(int));
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err=cudaMalloc((void **)&d_edgeArray, size2);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_vertexArray, Graph[0], size1, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_edgeArray, Graph[1], size2, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    int csize = (numVertices + 1) * sizeof(int);
    err = cudaMalloc((void **)&d_color, csize);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_color, color, csize, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    size_t pitch3;
    int *Queue;
    cudaMallocPitch((void**)&Queue, &pitch3, (numVertices + 1) * sizeof(bool), (numVertices + 1));
    CalculateAllFWBWSets<<<numBlocksPerGrid, numThreadsPerBlock>>>(FWReach, BWReach, pitch1, pitch2, d_color, d_vertexArray, d_edgeArray, Queue, pitch3, numVertices);
    cudaThreadSynchronize();
    
    for(int vertex = 1; vertex <= numVertices; vertex++) {
        outs("Forward Set for vertex: %d\n", vertex);
    
        err = cudaMemcpy(visited, ((char *)FWReach + vertex * pitch2), sizeBool, cudaMemcpyDeviceToHost);
        if( err != cudaSuccess)
        {
             printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
        }
        for(i = 1; i <= numVertices; i++) {
            outs("Vertex: %d - %d\n", i, visited[i]);
        }
    }
    for(int vertex = 1; vertex <= numVertices; vertex++) {
        outs("Backward Set for vertex: %d\n", vertex);
    
        err = cudaMemcpy(visited, ((char *)BWReach + vertex * pitch2), sizeBool, cudaMemcpyDeviceToHost);
        if( err != cudaSuccess)
        {
             printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
        }
        for(i = 1; i <= numVertices; i++) {
            outs("Vertex: %d - %d\n", i, visited[i]);
        }
    }
/*    
    for(int vertex = 1; vertex <= numVertices; vertex++) {
        print("Calculating FW Set for vertex: %d\n", vertex);
        visited[vertex] = true;
        CalculateFWSets(visited, color, vertex);
        err = cudaMemcpy(((char *)FWReach + vertex * pitch1), visited, sizeBool, cudaMemcpyHostToDevice);
        if( err != cudaSuccess)
        {
             printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
        }
        for(int i = 0; i <= numVertices; i++)
            visited[i] = false;
    
    }
    CheckReachVector<<<2, (numVertices+1)/2>>>(FWReach, pitch1, (numVertices + 1));
    for(int vertex = 1; vertex <= numVertices; vertex++) {
        print("Calculating BW Set for vertex: %d\n", vertex);
        visited[vertex] = true;
        CalculateBWSets(visited, color, vertex);
        err = cudaMemcpy(((char *)BWReach + vertex * pitch2), visited, sizeBool, cudaMemcpyHostToDevice);
        if( err != cudaSuccess)
        {
             printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
        }
        for(int i = 0; i <= numVertices; i++)
            visited[i] = false;
    
    }
    for(i = 0; i <= numVertices; i++) {
   //     visited[i] = false;
        color[i] = 0;
    }
    CheckReachVector<<<2, (numVertices+1)/2>>>(BWReach, pitch2, (numVertices + 1));

    err=cudaMalloc((void **)&d_vertexArray, (numVertices + 2) * sizeof(int *));
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err=cudaMalloc((void **)&d_edgeArray, size2);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_vertexArray, Graph[0], size1, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_edgeArray, Graph[1], size2, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
*/

    for(i = 0; i <= numVertices; i++) {
   //     visited[i] = false;
        color[i] = 0;
    }
    size = (numVertices + 1) * sizeof(int);
    err = cudaMalloc((void **)&d_SCC, size);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMalloc((void **)&d_mark, (numVertices + 1) * sizeof(bool));
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
/*    err = cudaMalloc((void **)&d_color, size);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }*/
    err = cudaMalloc((void **)&d_workQueue, 1000 * sizeof(int));
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMalloc((void **)&d_workQueue2, 1000 * sizeof(int));
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_SCC, SCC, size, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_mark, mark, (numVertices + 1) * sizeof(bool), cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
/*    err = cudaMemcpy(d_color, color, size, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }*/
    err = cudaMemcpy(d_workQueue, workQueue, 1000 * sizeof(int), cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_workQueue2, workQueue, 1000 * sizeof(int), cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    bool terminate = false;
    int currentQueueHead = 1, currentQueueTail = 0;
    cudaMemcpyToSymbol(d_queueHead, &currentQueueHead, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_queueTail, &currentQueueTail, sizeof(int), 0, cudaMemcpyHostToDevice);
    i = 0;
    while (terminate == false) {
        terminate = true;
        outs("Queue: Head: %d, Tail: %d\n", currentQueueHead, currentQueueTail);
        cudaMemcpyToSymbol(d_terminate, &terminate, sizeof(bool), 0, cudaMemcpyHostToDevice);
//      printf("Before Terminate on thread: %d, Host Terminate = %d\n", d_terminate, terminate);
        FWBWAlgo_Kernel<<<1, 500 >>>(d_SCC, d_color, d_mark, d_vertexArray, d_edgeArray, d_workQueue, d_workQueue2, numVertices/*, d_SubGraphColor*/, FWReach, BWReach, pitch1, pitch2, /*d_queueHead, d_queueTail,*/ currentQueueHead, currentQueueTail);
        print("Here\n");
        cudaThreadSynchronize();
        cudaMemcpyFromSymbol(&terminate, d_terminate, sizeof(bool), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&currentQueueHead, d_queueHead, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&currentQueueTail, d_queueTail, sizeof(int), 0, cudaMemcpyDeviceToHost);
        temp = currentQueueTail;
        cudaMemcpyToSymbol(d_queueHead, &temp, sizeof(int), 0, cudaMemcpyHostToDevice);
        temp = 0;
        cudaMemcpyToSymbol(d_queueTail, &temp, sizeof(int), 0, cudaMemcpyHostToDevice);
        cudaMemcpyFromSymbol(&currentQueueHead, d_queueHead, sizeof(int), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&currentQueueTail, d_queueTail, sizeof(int), 0, cudaMemcpyDeviceToHost);
//    outs("Thread Queue: Head: %d, Tail: %d\n", currentQueueHead, currentQueueTail);
        i++;
        int *temp2 = d_workQueue;
        d_workQueue = d_workQueue2;
        d_workQueue2 = temp2;
//        if (currentQueueHead >= 1000)
    }
    size = (numVertices + 1) * sizeof(int);
    err = cudaMemcpy(SCC, d_SCC, size, cudaMemcpyDeviceToHost);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    outs("\n\nSCC :: Iter = %d\n", i);
    for (int i = 1; i <= numVertices; i++) {
        outs("Color[%d] = %d\n", i, SCC[i]);
    }
    int c;
    for (int i = 1; i <= numVertices; i++) {
        if (SCC[i] == -1)
            continue;
        c = SCC[i];
        printf("%d ", i);
        for (int j = i+1; j <= numVertices; j++) {
            if (SCC[j] != -1 && SCC[j] == c) {
                printf("%d ", j);
                SCC[j] = -1;
            }
        }
        printf("\n");
    }
    
//    cudaFree(d_SubGraphColor);
    cudaFree(FWReach);
    cudaFree(BWReach);
    cudaFree(d_vertexArray);
    cudaFree(d_edgeArray);
    cudaFree(d_SCC);
    cudaFree(d_mark);
    cudaFree(d_color);
    cudaFree(d_workQueue);

    free(mark);
    free(workQueue);
    free(color);
    free(visited);
}
void CalculateFWSets(bool *visited, int *color, int vertex) {

    bool *d_visited;
    int i, j, size = (numVertices + 1) * sizeof(bool);
    j = Graph[0][vertex];
    /*for(j = Graph[0][vertex]; j < Graph[0][vertex + 1]; j++) {
        i = Graph[1][j];
        visited[i] = true;
    }*/
    int *d_vertexArray, *d_edgeArray, *d_color;
    int size1 = (numVertices + 2) * sizeof (int);
    int size2 = (numEdges + 1) * sizeof (int);
    cudaError_t err=cudaMalloc((void **)&d_vertexArray, (numVertices + 2) * sizeof(int));
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err=cudaMalloc((void **)&d_edgeArray, size2);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_vertexArray, Graph[0], size1, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_edgeArray, Graph[1], size2, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }

    err = cudaMalloc((void **)&d_visited, size);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_visited, visited, size, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    int csize = (numVertices + 1) * sizeof(int);
    err = cudaMalloc((void **)&d_color, csize);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_color, color, csize, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    bool terminate = false;
    print("Visited: Before\n");
    for(i = 1; i <= numVertices; i++) {
        print("Vertex: %d - %d\n", i, visited[i]);
    }
/*    print("Color: Before\n");
    for(i = 1; i <= numVertices; i++) {
        print("Vertex: %d - %d\n", i, color[i]);
    }*/
    while (terminate == false) {
       terminate = true;
        err = cudaMemcpy(d_visited, visited, size, cudaMemcpyHostToDevice);
        if( err != cudaSuccess)
        {
             printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
        }
        cudaMemcpyToSymbol(d_terminate, &terminate, sizeof(bool), 0, cudaMemcpyHostToDevice);
//        printf("Before Terminate on thread: %d, Host Terminate = %d\n", d_terminate, terminate);
    int numThreadsPerBlock = 256;
    int numBlocksPerGrid = (numVertices + 1 + numThreadsPerBlock - 1) / numThreadsPerBlock;
//        FW_Kernel<<<2, (numVertices + 1)/2 >>>(d_visited, d_color, d_vertexArray, d_edgeArray);
        FW_Kernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_visited, d_color, d_vertexArray, d_edgeArray, numVertices);
        cudaThreadSynchronize();
        cudaMemcpyFromSymbol(&terminate, d_terminate, sizeof(bool), 0, cudaMemcpyDeviceToHost);
        err = cudaMemcpy(visited, d_visited, size, cudaMemcpyDeviceToHost);
        if( err != cudaSuccess)
        {
             printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
        }
//    printf("Terminate on thread: %d, Host Terminate = %d\n", d_terminate, terminate);
        print("Visited: After\n");
        for(i = 1; i <= numVertices; i++) {
            print("Vertex: %d - %d\n", i, visited[i]);
        }
    }
    err = cudaMemcpy(visited, d_visited, size, cudaMemcpyDeviceToHost);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    
    cudaFree(d_visited);
    cudaFree(d_vertexArray);
    cudaFree(d_edgeArray);
    cudaFree(d_color);
}
void CalculateBWSets(bool *visited, int *color, int vertex) {

    bool *d_visited;
    int i, size = (numVertices + 1) * sizeof(bool);
    int *d_vertexArray, *d_edgeArray, *d_color;
    int size1 = (numVertices + 2) * sizeof (int);
    int size2 = (numEdges + 1) * sizeof (int);
    cudaError_t err=cudaMalloc((void **)&d_vertexArray, (numVertices + 2) * sizeof(int));
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err=cudaMalloc((void **)&d_edgeArray, size2);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_vertexArray, Graph[0], size1, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_edgeArray, Graph[1], size2, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }

    err = cudaMalloc((void **)&d_visited, size);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_visited, visited, size, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    int csize = (numVertices + 1) * sizeof(int);
    err = cudaMalloc((void **)&d_color, csize);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    err = cudaMemcpy(d_color, color, csize, cudaMemcpyHostToDevice);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    bool terminate = false;
    print("Visited: Before\n");
    for(i = 1; i <= numVertices; i++) {
        print("Vertex: %d - %d\n", i, visited[i]);
    }
    /*print("Color: Before\n");
    for(i = 1; i <= numVertices; i++) {
        print("Vertex: %d - %d\n", i, color[i]);
    }*/
    while (terminate == false) {
        terminate = true;
        err = cudaMemcpy(d_visited, visited, size, cudaMemcpyHostToDevice);
        if( err != cudaSuccess)
        {
             printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
        }
        cudaMemcpyToSymbol(d_terminate, &terminate, sizeof(bool), 0, cudaMemcpyHostToDevice);
//    printf("Before Terminate on thread: %d, Host Terminate = %d\n", d_terminate, terminate);
    int numThreadsPerBlock = 256;
    int numBlocksPerGrid = (numVertices + 1 + numThreadsPerBlock - 1) / numThreadsPerBlock;
    outs("Blockks: %d, Threads per block: %d\n", numBlocksPerGrid, numThreadsPerBlock);
//        BW_Kernel<<<2, (numVertices + 1)/2  >>>(d_visited, d_color, d_vertexArray, d_edgeArray);
//        BW_Kernel<<<1, 256  >>>(d_visited, d_color, d_vertexArray, d_edgeArray, numVertices);
        BW_Kernel<<<numBlocksPerGrid, numThreadsPerBlock>>>(d_visited, d_color, d_vertexArray, d_edgeArray, numVertices);
        cudaThreadSynchronize();
        cudaMemcpyFromSymbol(&terminate, d_terminate, sizeof(bool), 0, cudaMemcpyDeviceToHost);
        err = cudaMemcpy(visited, d_visited, size, cudaMemcpyDeviceToHost);
        if( err != cudaSuccess)
        {
             printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
        //     return EXIT_FAILURE;
        }
//    printf("Terminate on thread: %d, Host Terminate = %d\n", d_terminate, terminate);
        print("Visited:Vertex = %d. After\n", vertex);
        for(i = 1; i <= numVertices; i++) {
            print("Vertex: %d - %d\n", i, visited[i]);
        }
    }
    err = cudaMemcpy(visited, d_visited, size, cudaMemcpyDeviceToHost);
    if( err != cudaSuccess)
    {
         printf("CUDA error: %s ** at Line %d\n", cudaGetErrorString(err), __LINE__);
    //     return EXIT_FAILURE;
    }
    
    cudaFree(d_visited);
    cudaFree(d_vertexArray);
    cudaFree(d_edgeArray);
    cudaFree(d_color);
}
int main() {

    int **AdjMatrix;
    int i, j, k;

    scanf("%d %d", &numVertices, &numEdges);

// NumVertices starting from 1 to NumVertices plus an addition Sentinel node
// which points to the last index of the Graph[1] array.
    Graph[0] = (int *) malloc((numVertices + 2) * sizeof(int));
    Graph[1] = (int *) malloc((numEdges + 1) * sizeof(int));

    AdjMatrix = (int **) malloc((numVertices + 1) * sizeof(int *));
    for (i = 0; i <= numVertices; i++) {

        AdjMatrix[i] = (int *) malloc((numVertices + 1) * sizeof(int));
    }
    i = numEdges;
    int lastj = 1, currentIndex = 1;
    while(i) {

        scanf("%d %d", &j, &k);
        AdjMatrix[j][k] = 1;
        while (lastj <= j || lastj == 1) {
            if (lastj == 1) {
                Graph[0][0] = currentIndex;
                Graph[0][1] = currentIndex;
            }else {
                Graph[0][lastj] = currentIndex;
            }
            lastj++;
        }
//        if (AdjMatrix[k][j] != 1)
            Graph[1][currentIndex] = k;
        currentIndex ++;
        i--;
    }
    Graph[1][0] = 0;
    // Sentinel node just points to the end of the last node in the graph
    while (lastj <= numVertices + 1) {
        Graph[0][lastj] = currentIndex;
        lastj++;
    }
    //Graph[0][lastj+1] = currentIndex;
    for (i = 1; i <= numVertices + 1; i++) 
        print("Vertex: %d = %d\n", i, Graph[0][i]);

    print("Second Array:\n");
    for (i = 1; i <= numEdges; i++) 
        print("Edges: Index: %d, Value = %d\n", i, Graph[1][i]);

    j = 1;
    for (i = 1; i <= numVertices; i++) {

        currentIndex = Graph[0][i];
        while (currentIndex < Graph[0][i+1]) {
//            print("%d %d\n", i, Graph[1][currentIndex]);
            if (AdjMatrix[i][Graph[1][currentIndex]] != 1 /*&&
                AdjMatrix[Graph[1][currentIndex]][i] != 1*/) {
                outs("\n\nGraph Do not Match\n\n");
                break;
            }
            j++;
            currentIndex ++;
        }
    }
//    outs("Number of Edges: %d\n", j);

    int *SCC = (int *)malloc((numVertices + 1) * sizeof(int));
    for(int i = 1; i <= numVertices; i++)
        SCC[i] = 0;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Cuda name: %s, numVertices = %d\n", prop.name, numVertices);
    printf("Cuda Reg Per Block: %d\n", prop.regsPerBlock);
    printf("Cuda Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Cuda Total Global Mem: %d\n", prop.totalGlobalMem);
    printf("Cuda Shared Mem per Block: %d\n", prop.sharedMemPerBlock);
    printf("Cuda Max Grid Size: %d\n", prop.maxGridSize[0]);
//    numVertices = 800;
    calculateSCC(SCC);

    free(Graph[0]);
    free(Graph[1]);
    for (i = 1; i <= numVertices; i++) {
        free(AdjMatrix[i]);
    }
    free(AdjMatrix);
    free(SCC);
}


