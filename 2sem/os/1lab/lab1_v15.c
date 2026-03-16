/*
 * Лабораторная работа №1 (Тип 3, Вариант 15)
 * Процессы. Разделяемая память. Синхронизация.
 * Сортировка массива рекурсивным разделением
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <time.h>
#include <errno.h>
#include <signal.h>

#define SHM_NAME "/shm_lab1_v15"
#define SEM_NAME_PREFIX "/sem_lab1_v15"
#define THRESHOLD 1000  // Порог для локальной сортировки

// Параметры задачи для потомка
typedef struct {
    int left;             // Левая граница интервала
    int right;            // Правая граница интервала
    int depth;            // Текущая глубина рекурсии
    int child_sem_index;  // Индекс семафора для оповещения родителя
    pid_t child_pid;      // PID потомка (для отладки)
} TaskParams;

// Структура данных в разделяемой памяти
typedef struct {
    int *array;           // Указатель на массив данных
    int size;             // Размер массива
    int max_depth;        // Максимальная глубина рекурсии
    sem_t *semaphores;    // Массив семафоров для синхронизации
    TaskParams *tasks;    // Параметры задач для потомков
    int process_count;    // Счётчик созданных процессов
    int sem_count;        // Количество семафоров
} SharedMemory;

// Глобальные переменные
static SharedMemory *shared_mem = NULL;
static int shm_fd = -1;
static int g_process_count = 0;
static int g_max_depth = 0;

// Функция получения текущего времени в миллисекундах
static long long get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// Функция локальной сортировки (вставками)
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

// Функция слияния двух отсортированных половин
static void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    
    // Создаём временные массивы
    int *L = malloc(n1 * sizeof(int));
    int *R = malloc(n2 * sizeof(int));
    
    if (!L || !R) {
        free(L);
        free(R);
        return;
    }
    
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];
    
    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }
    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
    
    free(L);
    free(R);
}

// Функция для генерации уникального имени семафора
static void generate_sem_name(char *buffer, size_t size, pid_t pid, int index) {
    snprintf(buffer, size, "%s_%d_%d", SEM_NAME_PREFIX, pid, index);
}

// Рекурсивная функция сортировки с созданием процессов
static void process_sort(int left, int right, int depth, const char *parent_sem_name) {
    pid_t my_pid = getpid();
    long long start_time = get_time_ms();
    
    // Вывод трассировочной информации
    fprintf(stderr, "[PID:%d] Родитель PID:%d, Глубина:%d, Интервал:[%d,%d], размер:%d\n",
            my_pid, getppid(), depth, left, right, right - left + 1);
    fprintf(stderr, "[PID:%d] Начало сортировки, время:%.3f сек\n",
            my_pid, start_time / 1000.0);
    
    // 1. Проверка базовых условий
    if (left >= right) {
        if (parent_sem_name != NULL) {
            sem_t *parent_sem = sem_open(parent_sem_name, 0);
            if (parent_sem != SEM_FAILED) {
                sem_post(parent_sem);
                sem_close(parent_sem);
            }
        }
        fprintf(stderr, "[PID:%d] Завершение (базовый случай), время:%.3f сек\n",
                my_pid, get_time_ms() / 1000.0);
        return;
    }
    
    // 2. Если достигнута максимальная глубина или массив маленький
    if (depth >= g_max_depth || (right - left) < THRESHOLD) {
        local_sort(shared_mem->array, left, right);
        if (parent_sem_name != NULL) {
            sem_t *parent_sem = sem_open(parent_sem_name, 0);
            if (parent_sem != SEM_FAILED) {
                sem_post(parent_sem);
                sem_close(parent_sem);
            }
        }
        fprintf(stderr, "[PID:%d] Завершение (локальная сортировка), время:%.3f сек\n",
                my_pid, get_time_ms() / 1000.0);
        return;
    }
    
    // 3. Разделение массива
    int mid = left + (right - left) / 2;
    
    // 4. Создание семафоров для потомков
    char sem_name1[64], sem_name2[64];
    generate_sem_name(sem_name1, sizeof(sem_name1), my_pid, 0);
    generate_sem_name(sem_name2, sizeof(sem_name2), my_pid, 1);
    
    sem_t *sem1 = sem_open(sem_name1, O_CREAT | O_EXCL, 0644, 0);
    sem_t *sem2 = sem_open(sem_name2, O_CREAT | O_EXCL, 0644, 0);
    
    if (sem1 == SEM_FAILED || sem2 == SEM_FAILED) {
        fprintf(stderr, "[PID:%d] Ошибка создания семафоров: %s\n", my_pid, strerror(errno));
        if (sem1 != SEM_FAILED) {
            sem_close(sem1);
            sem_unlink(sem_name1);
        }
        if (sem2 != SEM_FAILED) {
            sem_close(sem2);
            sem_unlink(sem_name2);
        }
        exit(1);
    }
    
    // 5. Создание первого потомка
    pid_t pid1 = fork();
    if (pid1 < 0) {
        fprintf(stderr, "[PID:%d] Ошибка fork() для первого потомка: %s\n", my_pid, strerror(errno));
        sem_close(sem1);
        sem_close(sem2);
        sem_unlink(sem_name1);
        sem_unlink(sem_name2);
        exit(1);
    }
    
    if (pid1 == 0) {
        // Потомок 1
        __atomic_add_fetch(&shared_mem->process_count, 1, __ATOMIC_SEQ_CST);
        process_sort(left, mid, depth + 1, sem_name1);
        exit(0);
    }
    
    // 6. Создание второго потомка
    pid_t pid2 = fork();
    if (pid2 < 0) {
        fprintf(stderr, "[PID:%d] Ошибка fork() для второго потомка: %s\n", my_pid, strerror(errno));
        sem_close(sem1);
        sem_close(sem2);
        sem_unlink(sem_name1);
        sem_unlink(sem_name2);
        exit(1);
    }
    
    if (pid2 == 0) {
        // Потомок 2
        __atomic_add_fetch(&shared_mem->process_count, 1, __ATOMIC_SEQ_CST);
        process_sort(mid + 1, right, depth + 1, sem_name2);
        exit(0);
    }
    
    // 7. Ожидание завершения потомков через семафоры
    sem_wait(sem1);
    sem_wait(sem2);
    
    // 7.5 Слияние отсортированных половин
    merge(shared_mem->array, left, mid, right);
    
    // 8. Очистка семафоров
    sem_close(sem1);
    sem_close(sem2);
    sem_unlink(sem_name1);
    sem_unlink(sem_name2);
    
    // 9. Ожидание завершения процессов (для избежания зомби)
    waitpid(pid1, NULL, 0);
    waitpid(pid2, NULL, 0);
    
    // 10. Сигнал родителю (если есть)
    if (parent_sem_name != NULL) {
        sem_t *parent_sem = sem_open(parent_sem_name, 0);
        if (parent_sem != SEM_FAILED) {
            sem_post(parent_sem);
            sem_close(parent_sem);
        }
    }
    
    fprintf(stderr, "[PID:%d] Завершение сортировки, время:%.3f сек\n",
            my_pid, get_time_ms() / 1000.0);
}

// Функция инициализации разделяемой памяти
static int init_shared_memory(int size, int max_depth) {
    // Создаём разделяемую память
    shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR | O_EXCL, 0644);
    if (shm_fd == -1) {
        fprintf(stderr, "Ошибка shm_open: %s\n", strerror(errno));
        return -1;
    }
    
    // Вычисляем необходимый размер
    size_t array_size = size * sizeof(int);
    size_t shm_size = sizeof(SharedMemory) + array_size;
    
    // Устанавливаем размер разделяемой памяти
    if (ftruncate(shm_fd, shm_size) == -1) {
        fprintf(stderr, "Ошибка ftruncate: %s\n", strerror(errno));
        shm_unlink(SHM_NAME);
        close(shm_fd);
        return -1;
    }
    
    // Отображаем в адресное пространство
    shared_mem = mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shared_mem == MAP_FAILED) {
        fprintf(stderr, "Ошибка mmap: %s\n", strerror(errno));
        shm_unlink(SHM_NAME);
        close(shm_fd);
        return -1;
    }
    
    // Инициализируем структуру
    shared_mem->size = size;
    shared_mem->max_depth = max_depth;
    shared_mem->process_count = 1;  // Корневой процесс
    shared_mem->sem_count = 0;
    shared_mem->array = (int *)((char *)shared_mem + sizeof(SharedMemory));
    
    return 0;
}

// Функция очистки ресурсов
static void cleanup_shared_memory(void) {
    if (shared_mem != NULL && shared_mem != MAP_FAILED) {
        munmap(shared_mem, sizeof(SharedMemory) + shared_mem->size * sizeof(int));
        shared_mem = NULL;
    }
    if (shm_fd != -1) {
        close(shm_fd);
        shm_fd = -1;
    }
    shm_unlink(SHM_NAME);
}

// Функция проверки корректности сортировки
static int check_sort_result(void) {
    int *arr = shared_mem->array;
    int size = shared_mem->size;
    
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            fprintf(stderr, "Ошибка сортировки: arr[%d]=%d > arr[%d]=%d\n",
                    i, arr[i], i + 1, arr[i + 1]);
            return 0;
        }
    }
    return 1;
}

// Функция для сравнения с qsort
static int compare_int(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

// Функция проверки с эталоном
static int verify_with_qsort(int size) {
    // Создаём копию массива
    int *original = malloc(size * sizeof(int));
    if (!original) {
        fprintf(stderr, "Недостаточно памяти для проверки\n");
        return 0;
    }
    memcpy(original, shared_mem->array, size * sizeof(int));
    
    // Сортируем копию через qsort
    qsort(original, size, sizeof(int), compare_int);
    
    // Сравниваем
    int result = 1;
    for (int i = 0; i < size; i++) {
        if (shared_mem->array[i] != original[i]) {
            result = 0;
            break;
        }
    }
    
    free(original);
    return result;
}

// Обработчик сигнала для очистки
static void signal_handler(int sig) {
    fprintf(stderr, "\nПолучен сигнал %d, очистка...\n", sig);
    cleanup_shared_memory();
    exit(1);
}

int main(int argc, char *argv[]) {
    int size = 100000;      // Размер массива по умолчанию
    int max_depth = 3;      // Максимальная глубина по умолчанию
    
    // Установка обработчика сигнала
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Парсинг аргументов
    if (argc >= 2) {
        size = atoi(argv[1]);
        if (size <= 0) {
            fprintf(stderr, "Некорректный размер массива: %s\n", argv[1]);
            return 1;
        }
    }
    if (argc >= 3) {
        max_depth = atoi(argv[2]);
        if (max_depth < 0) {
            fprintf(stderr, "Некорректная глубина рекурсии: %s\n", argv[2]);
            return 1;
        }
    }
    
    g_max_depth = max_depth;
    
    // Вывод заголовка
    printf("========================================\n");
    printf("Лабораторная работа №1 (Тип 3, Вариант 15)\n");
    printf("Размер массива: %d\n", size);
    printf("Максимальная глубина: %d\n", max_depth);
    printf("========================================\n\n");
    
    // Инициализация разделяемой памяти
    if (init_shared_memory(size, max_depth) != 0) {
        fprintf(stderr, "Ошибка инициализации разделяемой памяти\n");
        return 1;
    }
    
    // Заполнение массива случайными числами
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        shared_mem->array[i] = rand() % 1000000;
    }
    
    fprintf(stderr, "Массив заполнен, запуск сортировки...\n\n");
    
    // Запуск таймера
    long long start_time = get_time_ms();
    
    // Запуск сортировки из корневого процесса
    process_sort(0, size - 1, 0, NULL);
    
    // Остановка таймера
    long long end_time = get_time_ms();
    double total_time = (end_time - start_time) / 1000.0;
    
    // Получение статистики
    int process_count = shared_mem->process_count;
    
    // Проверка результатов
    int sort_ok = check_sort_result();
    int verify_ok = verify_with_qsort(size);
    
    // Вывод результатов
    printf("\n========================================\n");
    printf("РЕЗУЛЬТАТЫ:\n");
    printf("========================================\n");
    printf("Общее время выполнения: %.3f сек\n", total_time);
    printf("Создано процессов: %d\n", process_count);
    printf("Проверка сортировки: %s\n", sort_ok ? "ОК" : "FAILED");
    printf("Сравнение с qsort: %s\n", verify_ok ? "ОК" : "FAILED");
    
    if (total_time > 0) {
        double speed = size / total_time;
        printf("Скорость: %.0f эл/сек\n", speed);
    }
    printf("========================================\n");
    
    // Очистка
    cleanup_shared_memory();
    
    return (sort_ok && verify_ok) ? 0 : 1;
}
