#include <iostream>
#include <omp.h>

int main()
{
    // Get number of available threads
    int max_threads = omp_get_max_threads();
    std::cout << "OpenMP is installed. Maximum threads available: " << max_threads << std::endl;

    // Test parallel execution
    std::cout << "Running parallel section with " << max_threads << " threads:" << std::endl;
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
#pragma omp critical
        {
            std::cout << "Hello from thread " << thread_id << std::endl;
        }
    }

    return 0;
}