// Nathan Gasc & Rémi La Fay
// to run usung gcc: gcc -o projet projet.c -mavx -lpthread -lm

#include <time.h>
#include <stdio.h>
#include <immintrin.h>
#include <sys/time.h> // for timing
#include <math.h>

double now(){ // copy from the class
    struct timeval t; double f_t;
    gettimeofday(&t, NULL);
    f_t = t.tv_usec; f_t = f_t/1000000.0; f_t +=t.tv_sec;
    return f_t;
}

// Exercice 1
// sequential version

double dist(float *a, float *b, int n){
    double d = 0;
    for(int i = 0; i < n; i++){
        d += sqrt(a[i]*a[i]*a[i]*a[i] + b[i]*b[i]*b[i]*b[i]);
    }
    return d;
}

// Exercice 2
// same thing but we use AVX assuming a and b are aligned and n is a multiple of 8
double dist_avx(float *a, float *b, int n){
    double d = 0;
    __m256 mm_a, mm_b, mm_d;
    
    for(int i = 0; i < n; i+=8){
        mm_a = _mm256_load_ps(a+i);
        mm_b = _mm256_load_ps(b+i);
        mm_a = _mm256_mul_ps(mm_a, mm_a);
        mm_b = _mm256_mul_ps(mm_b, mm_b); // (a^4 = a^2 * a^2)
        mm_d = _mm256_add_ps(_mm256_mul_ps(mm_a, mm_a), _mm256_mul_ps(mm_b, mm_b));
        mm_d = _mm256_sqrt_ps(mm_d);
        d += mm_d[0] + mm_d[1] + mm_d[2] + mm_d[3] + mm_d[4] + mm_d[5] + mm_d[6] + mm_d[7];
    }
    return d;
}

// Exercice 3
// same thing than ex2 but this time we don't assume that a and b are aligned and that n is a multiple of 8

double dist_avx_unaligned(float *a, float *b, int n){
    double d = 0;
    __m256 mm_a, mm_b, mm_d;
    
    for(int i = 0; i < n; i+=8){
        mm_a = _mm256_loadu_ps(a+i);
        mm_b = _mm256_loadu_ps(b+i);
        mm_a = _mm256_mul_ps(mm_a, mm_a);
        mm_b = _mm256_mul_ps(mm_b, mm_b); // (a^4 = a^2 * a^2)
        mm_d = _mm256_add_ps(_mm256_mul_ps(mm_a, mm_a), _mm256_mul_ps(mm_b, mm_b));
        mm_d = _mm256_sqrt_ps(mm_d);
        d += mm_d[0] + mm_d[1] + mm_d[2] + mm_d[3] + mm_d[4] + mm_d[5] + mm_d[6] + mm_d[7];
    }
    // we deal with the remaining elements
    for(int i = n - n%8; i < n; i++){
        d += sqrt(a[i]*a[i]*a[i]*a[i] + b[i]*b[i]*b[i]*b[i]);
    }
    return d;
}

// Exercice 3 bis
// same thing than ex3 but we split the job between multiple thread using pthreads

#include <pthread.h>
#define NTHREADS 8

typedef struct {
    float *a;
    float *b;
    int n;
    double *d; // a pointer to the global d
    pthread_mutex_t *mutex; // ensuring d is written by only one thread at a time
} dist_avx_unaligned_args;


void *dist_avx_unaligned_threaded_worker(void *arg){
    dist_avx_unaligned_args *args = (dist_avx_unaligned_args *)arg;
    double d = 0;
    __m256 mm_a, mm_b, mm_d;
    for(int i = 0; i < args->n; i+=8){
        mm_a = _mm256_loadu_ps(args->a+i);
        mm_b = _mm256_loadu_ps(args->b+i);
        mm_a = _mm256_mul_ps(mm_a, mm_a);
        mm_b = _mm256_mul_ps(mm_b, mm_b); // (a^4 = a^2 * a^2)
        mm_d = _mm256_add_ps(_mm256_mul_ps(mm_a, mm_a), _mm256_mul_ps(mm_b, mm_b));
        mm_d = _mm256_sqrt_ps(mm_d);
        d += mm_d[0] + mm_d[1] + mm_d[2] + mm_d[3] + mm_d[4] + mm_d[5] + mm_d[6] + mm_d[7];
    }
    // we deal with the remaining elements
    for(int i = args->n - args->n%8; i < args->n; i++){
        d += sqrt(args->a[i]*args->a[i]*args->a[i]*args->a[i] + args->b[i]*args->b[i]*args->b[i]*args->b[i]);
    }
    pthread_mutex_lock(args->mutex);
    *(args->d) += d;
    pthread_mutex_unlock(args->mutex);
}

double dist_avx_unaligned_threaded(float *a, float *b, int n){
    double d = 0; // global d

    pthread_t threads[NTHREADS];
    dist_avx_unaligned_args args[NTHREADS];

    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);

    for(int i = 0; i < NTHREADS; i++){
        // we split the job between the threads
        args[i].a = a + i*n/NTHREADS;
        args[i].b = b + i*n/NTHREADS;
        args[i].n = n/NTHREADS;
        args[i].d = &d;
        args[i].mutex = &mutex;
        pthread_create(&threads[i], NULL, dist_avx_unaligned_threaded_worker, &args[i]);
    }
    for(int i = 0; i < NTHREADS; i++){
        pthread_join(threads[i], NULL);
    }
    pthread_mutex_destroy(&mutex);
    return d;
}


// Exercice 4
// the main function that creates the vectors with random values between 0 and 1, calls the different versions of the dist function,
// compares the results and times the execution of each version
// it also display the total acceleration = time of the sequential version / time of the parallel version

int main(){
    int n = 100000000;
    float *a = (float *)aligned_alloc(32, n*sizeof(float));
    float *b = (float *)aligned_alloc(32, n*sizeof(float));

    for(int i = 0; i < n; i++){
        a[i] = (float)rand()/(float)RAND_MAX;
        b[i] = (float)rand()/(float)RAND_MAX;
    }

    double t1, t2, t3, t4, t5, t6, t7, t8;
    t1 = now();
    double d1 = dist(a, b, n);
    t2 = now();
    double d2 = dist_avx(a, b, n);
    t3 = now();
    double d3 = dist_avx_unaligned(a, b, n);
    t4 = now();
    double d4 = dist_avx_unaligned_threaded(a, b, n);
    t5 = now();

    printf("The computed distances (should be the same for all versions):\n");
    printf("d1 = %f\n", d1);
    printf("d2 = %f\n", d2);
    printf("d3 = %f\n", d3);
    printf("d4 = %f\n", d4);

    printf("Time of execution (in seconds):\n");
    printf("t1 (seq) = %f\n", t2-t1);
    printf("t2 (avx aligned) = %f\n", t3-t2);
    printf("t3 (avx unaligned) = %f\n", t4-t3);
    printf("t4 (avx + %d threads) = %f\n", NTHREADS, t5-t4);

    printf("acceleration avx = %f\n", (t2-t1)/(t3-t2));
    printf("acceleration avx threaded = %f\n", (t2-t1)/(t5-t4));

    free(a);
    free(b);
    return 0;
}

// Sample output:

// The computed distances (should be the same for all versions):
// d1 = 5447397.969626
// d2 = 5447397.969407
// d3 = 5447397.969407
// d4 = 5447397.969407
// Time of execution (in seconds):
// t1 (seq) = 0.439922
// t2 (avx aligned) = 0.015194
// t3 (avx unaligned) = 0.015059
// t4 (avx + 8 threads) = 0.005796
// acceleration avx = 28.953788
// acceleration avx threaded = 75.901563

// We observe a big variability between different runs