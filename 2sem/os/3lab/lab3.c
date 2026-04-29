/*
 * Лабораторная работа №3 (Тип 3, вариант Quick Sort)
 * Рекурсивная быстрая сортировка массива с деревом процессов.
 * Обмен данными — неименованные каналы (pipe), синхронизация — wait() и
 * корректное закрытие неиспользуемых концов pipe.
 *
 * Запуск: ./quicksort_proc [размер_массива] [макс_глубина_рекурсии]
 * Пример: ./quicksort_proc 20 3
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>

#define MIN_SIZE 11 /* по ТЗ размер > 10 */

/* ---------- Вспомогательные функции ввода-вывода в pipe ---------- */

static void read_all(int fd, void *buf, size_t n) {
    unsigned char *p = (unsigned char *)buf;
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, p + got, n - got);
        if (r < 0) {
            perror("read");
            _exit(1);
        }
        if (r == 0) {
            fprintf(stderr, "read: неожиданный EOF\n");
            _exit(1);
        }
        got += (size_t)r;
    }
}

static void write_all(int fd, const void *buf, size_t n) {
    const unsigned char *p = (const unsigned char *)buf;
    size_t sent = 0;
    while (sent < n) {
        ssize_t w = write(fd, p + sent, n - sent);
        if (w < 0) {
            perror("write");
            _exit(1);
        }
        sent += (size_t)w;
    }
}

/* Отправка результата дочернего узла в родительский pipe: n, forks_in_subtree, данные */
static void write_sorted_chunk(int fd, int n, unsigned long forks_below, const int *data) {
    write_all(fd, &n, sizeof n);
    write_all(fd, &forks_below, sizeof forks_below);
    if (n > 0) write_all(fd, data, (size_t)n * sizeof(int));
}

static void insertion_sort(int *a, int n) {
    for (int i = 1; i < n; i++) {
        int key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

/*
 * Схема Ломуто: опорный — последний элемент.
 * После вызова: a[0..p-1] < pivot, a[p] == pivot, a[p+1..n-1] >= pivot.
 */
static int partition_lomuto(int *a, int n) {
    if (n <= 1) return 0;
    int pivot = a[n - 1];
    int i = 0;
    for (int j = 0; j < n - 1; j++) {
        if (a[j] < pivot) {
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
            i++;
        }
    }
    int t = a[i];
    a[i] = a[n - 1];
    a[n - 1] = t;
    return i;
}

/*
 * Рекурсивная параллельная часть в текущем процессе.
 * Возвращает число вызовов fork() во всём поддереве (включая потомков).
 * Дочерние процессы, созданные здесь, передают свой результат по pipe и завершаются (_exit).
 */
static unsigned long parallel_qsort(int *buf, int n, int depth, int max_depth) {
    if (n <= 1) return 0;

    if (depth >= max_depth) {
        insertion_sort(buf, n);
        return 0;
    }

    int p = partition_lomuto(buf, n);
    int nL = p;
    int nR = n - p - 1;

    if (nL == 0 || nR == 0) {
        insertion_sort(buf, n);
        return 0;
    }

    int pl_to[2], pl_from[2], pr_to[2], pr_from[2];
    if (pipe(pl_to) != 0 || pipe(pl_from) != 0 || pipe(pr_to) != 0 ||
        pipe(pr_from) != 0) {
        perror("pipe");
        insertion_sort(buf, n);
        return 0;
    }

    pid_t left_pid = fork();
    if (left_pid < 0) {
        perror("fork");
        close(pl_to[0]);
        close(pl_to[1]);
        close(pl_from[0]);
        close(pl_from[1]);
        close(pr_to[0]);
        close(pr_to[1]);
        close(pr_from[0]);
        close(pr_from[1]);
        insertion_sort(buf, n);
        return 0;
    }

    if (left_pid == 0) {
        close(pl_to[1]);
        close(pl_from[0]);
        close(pr_to[0]);
        close(pr_to[1]);
        close(pr_from[0]);
        close(pr_from[1]);

        if (dup2(pl_to[0], STDIN_FILENO) < 0 || dup2(pl_from[1], STDOUT_FILENO) < 0) {
            perror("dup2");
            _exit(1);
        }
        close(pl_to[0]);
        close(pl_from[1]);

        int nn;
        if (read(STDIN_FILENO, &nn, sizeof nn) != (ssize_t)sizeof nn) {
            fprintf(stderr, "дочерний процесс: неверный заголовок\n");
            _exit(1);
        }
        int *b = (int *)malloc((size_t)nn * sizeof(int));
        if (!b) {
            perror("malloc");
            _exit(1);
        }
        read_all(STDIN_FILENO, b, (size_t)nn * sizeof(int));

        unsigned long sub = parallel_qsort(b, nn, depth + 1, max_depth);
        write_sorted_chunk(STDOUT_FILENO, nn, sub, b);
        free(b);
        _exit(0);
    }

    pid_t right_pid = fork();
    if (right_pid < 0) {
        perror("fork");
        close(pl_to[0]);
        close(pl_to[1]);
        close(pl_from[0]);
        close(pl_from[1]);
        close(pr_to[0]);
        close(pr_to[1]);
        close(pr_from[0]);
        close(pr_from[1]);
        waitpid(left_pid, NULL, 0);
        insertion_sort(buf, n);
        return 0;
    }

    if (right_pid == 0) {
        close(pr_to[1]);
        close(pr_from[0]);
        close(pl_to[0]);
        close(pl_to[1]);
        close(pl_from[0]);
        close(pl_from[1]);
        /* pr_to[0] и pr_from[1] нужны для dup2; не закрывать до переназначения */

        if (dup2(pr_to[0], STDIN_FILENO) < 0 || dup2(pr_from[1], STDOUT_FILENO) < 0) {
            perror("dup2");
            _exit(1);
        }
        close(pr_to[0]);
        close(pr_from[1]);

        int nn;
        if (read(STDIN_FILENO, &nn, sizeof nn) != (ssize_t)sizeof nn) {
            fprintf(stderr, "дочерний процесс: неверный заголовок\n");
            _exit(1);
        }
        int *b = (int *)malloc((size_t)nn * sizeof(int));
        if (!b) {
            perror("malloc");
            _exit(1);
        }
        read_all(STDIN_FILENO, b, (size_t)nn * sizeof(int));

        unsigned long sub = parallel_qsort(b, nn, depth + 1, max_depth);
        write_sorted_chunk(STDOUT_FILENO, nn, sub, b);
        free(b);
        _exit(0);
    }

    close(pl_to[0]);
    close(pl_from[1]);
    close(pr_to[0]);
    close(pr_from[1]);

    write_all(pl_to[1], &nL, sizeof nL);
    write_all(pl_to[1], buf, (size_t)nL * sizeof(int));
    close(pl_to[1]);

    write_all(pr_to[1], &nR, sizeof nR);
    write_all(pr_to[1], buf + p + 1, (size_t)nR * sizeof(int));
    close(pr_to[1]);

    int nLout;
    unsigned long forkL;
    read_all(pl_from[0], &nLout, sizeof nLout);
    read_all(pl_from[0], &forkL, sizeof forkL);
    int *Lpart = (int *)malloc((size_t)nLout * sizeof(int));
    if (!Lpart) {
        perror("malloc");
        waitpid(left_pid, NULL, 0);
        waitpid(right_pid, NULL, 0);
        close(pl_from[0]);
        close(pr_from[0]);
        exit(1);
    }
    if (nLout > 0) read_all(pl_from[0], Lpart, (size_t)nLout * sizeof(int));
    close(pl_from[0]);

    int nRout;
    unsigned long forkR;
    read_all(pr_from[0], &nRout, sizeof nRout);
    read_all(pr_from[0], &forkR, sizeof forkR);
    int *Rpart = (int *)malloc((size_t)nRout * sizeof(int));
    if (!Rpart) {
        perror("malloc");
        free(Lpart);
        waitpid(left_pid, NULL, 0);
        waitpid(right_pid, NULL, 0);
        close(pr_from[0]);
        exit(1);
    }
    if (nRout > 0) read_all(pr_from[0], Rpart, (size_t)nRout * sizeof(int));
    close(pr_from[0]);

    waitpid(left_pid, NULL, 0);
    waitpid(right_pid, NULL, 0);

    int pivot_val = buf[p];
    memcpy(buf, Lpart, (size_t)nLout * sizeof(int));
    buf[nLout] = pivot_val;
    memcpy(buf + nLout + 1, Rpart, (size_t)nRout * sizeof(int));

    free(Lpart);
    free(Rpart);

    return forkL + forkR + 2UL;
}

static int check_sorted(const int *a, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (a[i] > a[i + 1]) return 0;
    }
    return 1;
}

static int compare_int(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

static int verify_against_qsort(const int *a, int n) {
    int *cpy = (int *)malloc((size_t)n * sizeof(int));
    if (!cpy) return 0;
    memcpy(cpy, a, (size_t)n * sizeof(int));
    qsort(cpy, (size_t)n, sizeof(int), compare_int);
    int ok = (memcmp(a, cpy, (size_t)n * sizeof(int)) == 0);
    free(cpy);
    return ok;
}

static void print_array(const char *title, const int *a, int n) {
    printf("%s", title);
    for (int i = 0; i < n; i++) printf("%d ", a[i]);
    printf("\n");
}

int main(int argc, char *argv[]) {
    int n = 20;
    int max_depth = 3;

    if (argc >= 2) {
        n = atoi(argv[1]);
        if (n < MIN_SIZE) {
            fprintf(stderr, "Размер массива должен быть не меньше %d\n", MIN_SIZE);
            return 1;
        }
    }
    if (argc >= 3) {
        max_depth = atoi(argv[2]);
        if (max_depth < 0) {
            fprintf(stderr, "Глубина не может быть отрицательной\n");
            return 1;
        }
    }

    printf("========================================\n");
    printf("Лабораторная работа №3: Quick Sort + pipe\n");
    printf("Размер массива:     %d\n", n);
    printf("Макс. глубина ветвления процессов: %d\n", max_depth);
    printf("========================================\n\n");

    int *arr = (int *)malloc((size_t)n * sizeof(int));
    if (!arr) {
        perror("malloc");
        return 1;
    }

    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) arr[i] = rand() % 1000;

    print_array("Исходный массив:\n", arr, n);

    unsigned long total_forks = parallel_qsort(arr, n, 0, max_depth);

    print_array("\nОтсортированный массив:\n", arr, n);

    int ok_order = check_sorted(arr, n);
    int ok_qsort = verify_against_qsort(arr, n);

    printf("\n========================================\n");
    printf("Проверка порядка:        %s\n", ok_order ? "ОК" : "ОШИБКА");
    printf("Сравнение с qsort:       %s\n", ok_qsort ? "ОК" : "ОШИБКА");
    printf("Всего вызовов fork() в дереве: %lu\n", total_forks);
    printf("(Каждый внутренний узел с двумя детьми даёт +2 к счётчику.)\n");
    printf("========================================\n");

    free(arr);
    return (ok_order && ok_qsort) ? 0 : 1;
}
