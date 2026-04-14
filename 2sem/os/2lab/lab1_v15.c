/*
 * Лабораторная работа №2 (Тип 3, Вариант 15)
 * Потоки (pthread). Синхронизация. Рекурсивная сортировка слиянием.
 *
 * Алгоритм:
 *  - Первый рекурсивный вызов выполняется в том же потоке.
 *  - Второй вызов — в зависимости от счётчика активных потоков:
 *    > max_threads  → в том же потоке
 *    <= max_threads → создаётся новый поток, ожидается через семафор.
 *  - Счётчик потоков защищён мьютексом.
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <errno.h>
#include <signal.h>

#define THRESHOLD 1000  /* Порог для локальной сортировки */

/* ---------- Глобальные данные ---------- */

static int *g_array = NULL;       /* сортируемый массив        */
static int   g_size = 0;          /* его размер                */
static int   g_max_threads = 0;   /* ограничение на число потоков (0 = 1 поток) */

static pthread_mutex_t g_thread_cnt_mtx = PTHREAD_MUTEX_INITIALIZER;
static int             g_active_threads = 0;   /* защищён мьютексом выше */

/* ---------- Утилиты ---------- */

static long long get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* Вставками — для маленьких подмассивов */
static void local_sort(int arr[], int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= left && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

/* Слияние двух отсортированных половин */
static void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = malloc(n1 * sizeof(int));
    int *R = malloc(n2 * sizeof(int));
    if (!L || !R) { free(L); free(R); return; }

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int j = 0; j < n2; j++) R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

/* ---------- Параметры для потока ---------- */

typedef struct {
    int  left;
    int  right;
    int  depth;
    sem_t *sem;       /* семафор для сигнала завершения (NULL у корня) */
} ThreadArgs;

/* ---------- Прямое объявление ---------- */

static void thread_sort(int left, int right, int depth, sem_t *parent_sem);

/* ---------- Точка входа потока ---------- */

static void *thread_entry(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    thread_sort(args->left, args->right, args->depth, args->sem);
    free(args);
    return NULL;
}

/* ---------- Основная рекурсивная функция ---------- */

static void thread_sort(int left, int right, int depth, sem_t *parent_sem) {

    fprintf(stderr, "[TID:0x%lx] Глубина:%d  Интервал:[%d,%d]  размер:%d\n",
            (unsigned long)pthread_self(), depth, left, right, right - left + 1);

    /* Базовый случай */
    if (left >= right) {
        if (parent_sem) sem_post(parent_sem);
        return;
    }

    /* Локальная сортировка, если порог превышен или глубина исчерпана */
    if (depth >= g_max_threads || (right - left) < THRESHOLD) {
        local_sort(g_array, left, right);
        if (parent_sem) sem_post(parent_sem);
        return;
    }

    int mid = left + (right - left) / 2;

    /* --- Определяем, стоит ли создавать новый поток --- */
    int should_spawn = 0;

    pthread_mutex_lock(&g_thread_cnt_mtx);
    if (g_active_threads < g_max_threads) {
        g_active_threads++;
        should_spawn = 1;
    }
    pthread_mutex_unlock(&g_thread_cnt_mtx);

    if (should_spawn) {
        /* Создаём семафор для синхронизации с новым потоком */
        sem_t *child_sem = malloc(sizeof(sem_t));
        if (!child_sem) {
            perror("malloc sem");
            exit(1);
        }
        if (sem_init(child_sem, 0, 0) != 0) {
            perror("sem_init");
            free(child_sem);
            exit(1);
        }

        /* Формируем аргументы для нового потока */
        ThreadArgs *targs = malloc(sizeof(ThreadArgs));
        if (!targs) {
            perror("malloc ThreadArgs");
            sem_destroy(child_sem);
            free(child_sem);
            exit(1);
        }
        targs->left  = mid + 1;
        targs->right = right;
        targs->depth = depth + 1;
        targs->sem   = child_sem;

        pthread_t tid;
        if (pthread_create(&tid, NULL, thread_entry, targs) != 0) {
            perror("pthread_create");
            free(targs);
            sem_destroy(child_sem);
            free(child_sem);
            /* Откатываем счётчик */
            pthread_mutex_lock(&g_thread_cnt_mtx);
            g_active_threads--;
            pthread_mutex_unlock(&g_thread_cnt_mtx);
            /* Фолбэк: сортируем последовательно */
            thread_sort(mid + 1, right, depth + 1, NULL);
        } else {
            /* Первый вызов — в текущем потоке */
            thread_sort(left, mid, depth + 1, NULL);

            /* Ожидаем завершения дочернего потока */
            sem_wait(child_sem);
            pthread_join(tid, NULL);

            sem_destroy(child_sem);
            free(child_sem);

            /* Декремент счётчика */
            pthread_mutex_lock(&g_thread_cnt_mtx);
            g_active_threads--;
            pthread_mutex_unlock(&g_thread_cnt_mtx);
        }
    } else {
        /* Превышен лимит — оба вызова в том же потоке */
        thread_sort(left, mid, depth + 1, NULL);
        thread_sort(mid + 1, right, depth + 1, NULL);
    }

    /* Слияние */
    merge(g_array, left, mid, right);

    /* Сигнал родителю */
    if (parent_sem) sem_post(parent_sem);
}

/* ---------- Проверка сортировки ---------- */

static int check_sort(void) {
    for (int i = 0; i < g_size - 1; i++)
        if (g_array[i] > g_array[i + 1]) {
            fprintf(stderr, "Ошибка: arr[%d]=%d > arr[%d]=%d\n",
                    i, g_array[i], i + 1, g_array[i + 1]);
            return 0;
        }
    return 1;
}

static int compare_int(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

static int verify_with_qsort(void) {
    int *cpy = malloc(g_size * sizeof(int));
    if (!cpy) return 0;
    memcpy(cpy, g_array, g_size * sizeof(int));
    qsort(cpy, g_size, sizeof(int), compare_int);
    int ok = 1;
    for (int i = 0; i < g_size; i++)
        if (g_array[i] != cpy[i]) { ok = 0; break; }
    free(cpy);
    return ok;
}

/* ---------- main ---------- */

int main(int argc, char *argv[]) {
    g_size        = 100000;   /* по умолчанию         */
    g_max_threads = 4;        /* максимальное кол-во потоков */

    if (argc >= 2) {
        g_size = atoi(argv[1]);
        if (g_size <= 0) { fprintf(stderr, "Некорректный размер\n"); return 1; }
    }
    if (argc >= 3) {
        g_max_threads = atoi(argv[2]);
        if (g_max_threads < 0) { fprintf(stderr, "Некорректное кол-во потоков\n"); return 1; }
    }

    printf("========================================\n");
    printf("Лабораторная работа №2 (Тип 3, Вариант 15)\n");
    printf("Потоки (pthread)\n");
    printf("Размер массива:  %d\n", g_size);
    printf("Макс. потоков:   %d\n", g_max_threads);
    printf("========================================\n\n");

    /* Выделяем массив в обычной памяти */
    g_array = malloc(g_size * sizeof(int));
    if (!g_array) {
        perror("malloc");
        return 1;
    }

    /* Заполняем случайными числами */
    srand((unsigned)time(NULL));
    for (int i = 0; i < g_size; i++)
        g_array[i] = rand() % 1000000;

    fprintf(stderr, "Массив заполнен, запуск сортировки...\n\n");

    long long start = get_time_ms();

    /* g_max_threads == 0  →  один поток (без распараллеливания) */
    if (g_max_threads == 0) {
        thread_sort(0, g_size - 1, 0, NULL);
    } else {
        thread_sort(0, g_size - 1, 0, NULL);
    }

    long long end  = get_time_ms();
    double  total  = (end - start) / 1000.0;

    /* Проверки */
    int sort_ok  = check_sort();
    int verify_ok = verify_with_qsort();

    printf("\n========================================\n");
    printf("РЕЗУЛЬТАТЫ:\n");
    printf("========================================\n");
    printf("Общее время:      %.3f сек\n", total);
    printf("Макс. потоков:    %d\n", g_max_threads);
    printf("Сортировка верна: %s\n", sort_ok ? "ОК" : "FAILED");
    printf("Сравнение с qsort: %s\n", verify_ok ? "ОК" : "FAILED");
    if (total > 0)
        printf("Скорость:         %.0f эл/сек\n", g_size / total);
    printf("========================================\n");

    free(g_array);
    pthread_mutex_destroy(&g_thread_cnt_mtx);

    return (sort_ok && verify_ok) ? 0 : 1;
}
