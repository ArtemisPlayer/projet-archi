// Nathan Gasc & Rémi La Fay
// Pour compiler: gcc -o projet projet.c -mavx -lpthread -lm

#include <time.h>
#include <stdio.h>
#include <immintrin.h>
#include <sys/time.h>
#include <math.h>

double now(){
    struct timeval t; double f_t;
    gettimeofday(&t, NULL);
    f_t = t.tv_usec; f_t = f_t/1000000.0; f_t +=t.tv_sec;
    return f_t;
}

// Exercice 1
// Version séquentielle 

double dist(float *a, float *b, int n){
    double d = 0;
    for(int i = 0; i < n; i++){
        d += sqrt(a[i]*a[i]*a[i]*a[i] + b[i]*b[i]*b[i]*b[i]);
    }
    return d;
}

// Exercice 2
// Version vectorisée avec AVX. On suppose que a et b sont alignés et 
// que n est un multiple de 8
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
// Version vectorisée avec AVX. On suppose que a et b ne sont pas nécessairement 
// alignés et que n n'est pas nécessairement un multiple de 8

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
    // On traite les éléments restants
    for(int i = n - n%8; i < n; i++){
        d += sqrt(a[i]*a[i]*a[i]*a[i] + b[i]*b[i]*b[i]*b[i]);
    }
    return d;
}

// Exercice 3 bis
// Idem que l'exercice 3 mais on répartit le travail entre plusieurs threads

#include <pthread.h>
#define NTHREADS 8

typedef struct {
    float *a;
    float *b;
    int n;
    double *d; // un pointeur vers la variable globale d
    pthread_mutex_t *mutex; // pour protéger l'accès à d
} dist_avx_unaligned_args;

// fonction exécutée par chaque thread
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
    for(int i = args->n - args->n%8; i < args->n; i++){
        d += sqrt(args->a[i]*args->a[i]*args->a[i]*args->a[i] + args->b[i]*args->b[i]*args->b[i]*args->b[i]);
    }
    pthread_mutex_lock(args->mutex);
    *(args->d) += d;
    pthread_mutex_unlock(args->mutex);
}

// fonction principale
double dist_avx_unaligned_threaded(float *a, float *b, int n){
    double d = 0; // global d

    pthread_t threads[NTHREADS];
    dist_avx_unaligned_args args[NTHREADS];

    pthread_mutex_t mutex;
    pthread_mutex_init(&mutex, NULL);

    for(int i = 0; i < NTHREADS; i++){
        // on sépare le travail en NTHREADS parties
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
// La fonction main crée les vecteurs avec des valeurs aléatoires entre 0 et 1, 
// appelle les différentes versions de la fonction dist,
// compare les résultats et mesure le temps d'exécution de chaque version et
// elle affiche aussi l'accélération totale = temps de la version séquentielle / temps de la version parallèle

// Exemple d'output:

// Les distances calculées (doivent être identiques):
// d1 = 54467744.926057
// d2 = 54467744.928190
// d3 = 54467744.928190
// d4 = 54467744.928190
// Temps d'exécution en seconde:
// t1 (seq) = 3.623515
// t2 (avx aligned) = 0.151453
// t3 (avx unaligned) = 0.149246
// t4 (avx + 8 threads) = 0.033610
// acceleration avx = 23.925011
// acceleration avx threaded = 107.811052

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

    printf("Les distances calculées (doivent être identiques):\n");
    printf("d1 = %f\n", d1);
    printf("d2 = %f\n", d2);
    printf("d3 = %f\n", d3);
    printf("d4 = %f\n", d4);

    printf("Temps d'exécution en seconde:\n");
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
