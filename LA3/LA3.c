#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <omp.h>

double randomValue(unsigned int* seed);
bool throwDart(unsigned int* seed);

int main(int argc, char** argv) {

    if (argc != 3) {
        printf("Error: there must be 2 arguments\n");
        return 1;
    }

    // Max number of threads that can be used based on the system and runtime
    int maxThreads = omp_get_max_threads();

    // N is num darts, should not be more than 2.14 billion
    int N = atoi(argv[1]);
    printf("N is %d\n", N);

    // Can't be more than maxThreads
    int T = atoi(argv[2]);
    if (T > maxThreads) {
        printf("The supplied number of threads (%d) can not be more than %d", T, maxThreads);
        return 1;
    }
    printf("T is %d\n", T);

    // H is the number of hits
    int H = 0;

    // Start the timer
    double startTime = omp_get_wtime();

    int timeSeed = time(NULL);

    // I think reduction would be faster than using atomic here since we update it billions of times. I have not yet tested
    // atomic speed vs reduction speed *after* realizing that the mutex was slowing it down, but reduction was definitely faster before
    #pragma omp parallel num_threads(T)
    {
        // I found that if we create seed outside of personal-thread-context (i.e: above this pragma), it takes insanely 
        // longer the more threads we use. I think this is because the OMP must be applying some type of mutex lock
        // onto the variable if declared outside, so the threads block each other from accessing it creating
        // a ton of overhead. On the contrary, I've noticed that declaring the seed within each iteration of the loop
        // creates major inaccuracies, not to mention that adds unneccessary overhead too (declaring seed 2 billion times)
        unsigned int seed = timeSeed + omp_get_thread_num();
        
        #pragma omp for reduction(+:H)
        for (int i = 0; i < N; i++) {
            if (throwDart(&seed)) {
                H++;
            }
        }
    }

    printf("H is %d\n", H);

    double pi = acos(-1.0);
    printf("pi is %.12f\n", pi);

    double piApprox = 4*((double)H/(double)N);
    printf("piApprox is %.12f\n", piApprox);

    double error = fabs((pi - piApprox) / pi) * 100;
    printf("Percentage Error: %.12f%%\n", error);

    printf("Total time: %f seconds\n", omp_get_wtime() - startTime);
}

bool throwDart(unsigned int* seed) {

    double x = randomValue(seed);
    double y = randomValue(seed);

    return x*x + y*y <= 1;
}

// Returns a value between [-1, 1]
double randomValue(unsigned int* seed) {

    // Generate a random value between 0 and RAND_MAX (seems to be same as int32 max, 2.14 billion ish)
    int randomValue = rand_r(seed);

    // Normalize the value so it is between 0 and 1
    double normalizedValue = (double)randomValue / (double)RAND_MAX;
    
    // Scale the value so it is between 0 and 2, then shift it down by 1 so it is between -1 and 1
    return normalizedValue * 2.0 - 1.0;
}