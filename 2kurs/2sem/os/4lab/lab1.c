/*
 * Лабораторная работа: задание лаб. 1 на TCP-сокетах.
 * Сервер: 127.0.0.1, предпочтительный порт = младшие 16 бит PID; при EADDRINUSE
 * перебираются следующие порты. Фактический порт передаётся потомкам в LAB1_SRV_PORT.
 * Потомок подключается к 127.0.0.1 и читает порт из LAB1_SRV_PORT (или PID mod 65536).
 * Родитель -> потомок: int32 глубина_рекурсии+1, int32 n, n элементов int32.
 * Потомок -> родитель: int32 счётчик_обменов, int32 n, n элементов int32.
 */

#define _POSIX_C_SOURCE 200809L

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define BACKLOG 8
#define LOCALHOST "127.0.0.1"
#define ENV_SRV_PORT "LAB1_SRV_PORT"
#define MAX_PORT_FALLBACK 4096u

static int g_max_depth = 3;

static uint16_t pid_to_port(pid_t p) {
    return (uint16_t)((unsigned long)p & 0xFFFFu);
}

/* bind + listen на одном порту; при ошибке сокет закрыт, errno сохранён */
static int bind_listen_port(uint16_t port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("socket");
        return -1;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, LOCALHOST, &addr.sin_addr) != 1) {
        fputs("inet_pton failed\n", stderr);
        close(fd);
        errno = EINVAL;
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        int saved = errno;
        close(fd);
        errno = saved;
        return -1;
    }
    if (listen(fd, BACKLOG) < 0) {
        perror("listen");
        close(fd);
        return -1;
    }
    return fd;
}

/*
 * Сначала pid_to_port(owner), затем +1, +2, … пока bind не примет EADDRINUSE как повод
 * попробовать следующий порт. Иные ошибки bind/listen — сразу выход с -1.
 */
static int make_server_with_fallback(pid_t owner, uint16_t *chosen_port) {
    uint16_t base = pid_to_port(owner);
    for (unsigned attempt = 0; attempt < MAX_PORT_FALLBACK; attempt++) {
        uint16_t port = (uint16_t)(base + (uint16_t)attempt);
        int fd = bind_listen_port(port);
        if (fd >= 0) {
            *chosen_port = port;
            if (attempt > 0)
                fprintf(stderr,
                        "предпочтительный порт %u занят, слушаем запасной %u (pid %jd)\n",
                        (unsigned)base, (unsigned)port, (intmax_t)owner);
            return fd;
        }
        if (errno != EADDRINUSE) {
            fprintf(stderr, "bind 127.0.0.1:%u (pid %jd): %s\n", (unsigned)port,
                    (intmax_t)owner, strerror(errno));
            return -1;
        }
    }
    fprintf(stderr,
            "не удалось занять порт: перебрано %u вариантов начиная с %u (pid %jd)\n",
            (unsigned)MAX_PORT_FALLBACK, (unsigned)base, (intmax_t)owner);
    return -1;
}

static void publish_listen_port(uint16_t port) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%u", (unsigned)port);
    if (setenv(ENV_SRV_PORT, buf, 1) != 0)
        perror("setenv " ENV_SRV_PORT);
}

static uint16_t port_for_connect(pid_t parent_pid) {
    const char *env = getenv(ENV_SRV_PORT);
    if (env && env[0]) {
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0' && v > 0ul && v <= 65535ul)
            return (uint16_t)v;
    }
    return pid_to_port(parent_pid);
}

static int connect_to_parent(pid_t parent_pid) {
    uint16_t port = port_for_connect(parent_pid);
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("socket (client)");
        return -1;
    }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, LOCALHOST, &addr.sin_addr) != 1) {
        close(fd);
        return -1;
    }
    for (int attempt = 0; attempt < 8000; attempt++) {
        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0)
            return fd;
        if (errno != ECONNREFUSED) {
            fprintf(stderr, "connect 127.0.0.1:%u: %s\n", (unsigned)port, strerror(errno));
            close(fd);
            return -1;
        }
        struct timespec ts = {0, 500000};
        nanosleep(&ts, NULL);
    }
    fprintf(stderr, "connect: таймаут 127.0.0.1:%u\n", (unsigned)port);
    close(fd);
    return -1;
}

static int write_full(int fd, const void *buf, size_t len) {
    const char *p = (const char *)buf;
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, p + off, len - off);
        if (w < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (w == 0)
            return -1;
        off += (size_t)w;
    }
    return 0;
}

static int read_full(int fd, void *buf, size_t len) {
    char *p = (char *)buf;
    size_t off = 0;
    while (off < len) {
        ssize_t r = read(fd, p + off, len - off);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (r == 0)
            return -1;
        off += (size_t)r;
    }
    return 0;
}

static int accept_conn(int serv) {
    int c = accept(serv, NULL, NULL);
    if (c < 0)
        perror("accept");
    return c;
}

/* Сортировка вставками; счётчик — число сдвигов при вставке элемента. */
static int local_sort_count_swaps(int32_t *a, int n) {
    int swaps = 0;
    for (int i = 1; i < n; i++) {
        int32_t key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            swaps++;
            j--;
        }
        a[j + 1] = key;
    }
    return swaps;
}

static void merge_halves(int32_t *dst, const int32_t *L, int n1, const int32_t *R, int n2) {
    int i = 0, j = 0, k = 0;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            dst[k++] = L[i++];
        else
            dst[k++] = R[j++];
    }
    while (i < n1)
        dst[k++] = L[i++];
    while (j < n2)
        dst[k++] = R[j++];
}

static void sort_with_upstream(int upstream, int32_t depth_next, int32_t *arr, int32_t n);

/* Потомок: подключился к родителю, читает задание и отдаёт результат на том же сокете. */
static void worker_main(void) {
    int peer = connect_to_parent(getppid());
    if (peer < 0)
        _exit(1);
    int32_t depth_next = 0, n = 0;
    if (read_full(peer, &depth_next, sizeof(depth_next)) < 0 ||
        read_full(peer, &n, sizeof(n)) < 0 || n < 0 || n > 100000000) {
        close(peer);
        _exit(1);
    }
    int32_t *buf = malloc((size_t)n * sizeof(int32_t));
    if (!buf) {
        close(peer);
        _exit(1);
    }
    if (n > 0 && read_full(peer, buf, (size_t)n * sizeof(int32_t)) < 0) {
        free(buf);
        close(peer);
        _exit(1);
    }
    sort_with_upstream(peer, depth_next, buf, n);
    free(buf);
    close(peer);
    _exit(0);
}

/*
 * Сортировка сегмента arr[0..n-1]; depth_next — «глубина рекурсии+1» от родителя.
 * Если upstream >= 0 — отправить наверх swaps, n, arr.
 */
static void sort_with_upstream(int upstream, int32_t depth_next, int32_t *arr, int32_t n) {
    if (n <= 1) {
        int32_t sw = 0;
        if (upstream >= 0) {
            if (write_full(upstream, &sw, sizeof(sw)) < 0 || write_full(upstream, &n, sizeof(n)) < 0 ||
                (n > 0 && write_full(upstream, arr, (size_t)n * sizeof(int32_t)) < 0))
                _exit(1);
        }
        return;
    }

    if (depth_next >= g_max_depth) {
        int32_t sw = local_sort_count_swaps(arr, n);
        if (upstream >= 0) {
            if (write_full(upstream, &sw, sizeof(sw)) < 0 || write_full(upstream, &n, sizeof(n)) < 0 ||
                write_full(upstream, arr, (size_t)n * sizeof(int32_t)) < 0)
                _exit(1);
        }
        return;
    }

    int mid = n / 2;
    int32_t n1 = mid;
    int32_t n2 = n - mid;

    uint16_t chosen = 0;
    int serv = make_server_with_fallback(getpid(), &chosen);
    if (serv < 0)
        _exit(1);
    publish_listen_port(chosen);

    pid_t c1 = fork();
    if (c1 < 0) {
        perror("fork");
        close(serv);
        _exit(1);
    }
    if (c1 == 0) {
        close(serv);
        worker_main();
    }

    int a1 = accept_conn(serv);
    if (a1 < 0) {
        close(serv);
        kill(c1, SIGTERM);
        waitpid(c1, NULL, 0);
        _exit(1);
    }

    int32_t dch = depth_next + 1;
    if (write_full(a1, &dch, sizeof(dch)) < 0 || write_full(a1, &n1, sizeof(n1)) < 0 ||
        write_full(a1, arr, (size_t)n1 * sizeof(int32_t)) < 0) {
        close(a1);
        close(serv);
        kill(c1, SIGTERM);
        waitpid(c1, NULL, 0);
        _exit(1);
    }

    pid_t c2 = fork();
    if (c2 < 0) {
        perror("fork");
        close(a1);
        close(serv);
        kill(c1, SIGTERM);
        waitpid(c1, NULL, 0);
        _exit(1);
    }
    if (c2 == 0) {
        close(serv);
        close(a1);
        worker_main();
    }

    int a2 = accept_conn(serv);
    close(serv);
    if (a2 < 0) {
        close(a1);
        kill(c1, SIGTERM);
        kill(c2, SIGTERM);
        waitpid(c1, NULL, 0);
        waitpid(c2, NULL, 0);
        _exit(1);
    }

    if (write_full(a2, &dch, sizeof(dch)) < 0 || write_full(a2, &n2, sizeof(n2)) < 0 ||
        write_full(a2, arr + n1, (size_t)n2 * sizeof(int32_t)) < 0) {
        close(a1);
        close(a2);
        kill(c1, SIGTERM);
        kill(c2, SIGTERM);
        waitpid(c1, NULL, 0);
        waitpid(c2, NULL, 0);
        _exit(1);
    }

    int32_t sw1 = 0, sw2 = 0, nn1 = 0, nn2 = 0;
    int32_t *L = malloc((size_t)n1 * sizeof(int32_t));
    int32_t *R = malloc((size_t)n2 * sizeof(int32_t));
    if (!L || !R) {
        free(L);
        free(R);
        close(a1);
        close(a2);
        kill(c1, SIGTERM);
        kill(c2, SIGTERM);
        waitpid(c1, NULL, 0);
        waitpid(c2, NULL, 0);
        _exit(1);
    }

    if (read_full(a1, &sw1, sizeof(sw1)) < 0 || read_full(a1, &nn1, sizeof(nn1)) < 0 || nn1 != n1 ||
        read_full(a1, L, (size_t)n1 * sizeof(int32_t)) < 0 || read_full(a2, &sw2, sizeof(sw2)) < 0 ||
        read_full(a2, &nn2, sizeof(nn2)) < 0 || nn2 != n2 ||
        read_full(a2, R, (size_t)n2 * sizeof(int32_t)) < 0) {
        free(L);
        free(R);
        close(a1);
        close(a2);
        kill(c1, SIGTERM);
        kill(c2, SIGTERM);
        waitpid(c1, NULL, 0);
        waitpid(c2, NULL, 0);
        _exit(1);
    }

    close(a1);
    close(a2);
    waitpid(c1, NULL, 0);
    waitpid(c2, NULL, 0);

    merge_halves(arr, L, n1, R, n2);
    free(L);
    free(R);

    int32_t sw = sw1 + sw2;
    if (upstream >= 0) {
        if (write_full(upstream, &sw, sizeof(sw)) < 0 || write_full(upstream, &n, sizeof(n)) < 0 ||
            write_full(upstream, arr, (size_t)n * sizeof(int32_t)) < 0)
            _exit(1);
    }
}

/* Корень: глубина рекурсии 0 -> потомкам уходит depth_next = 1. */
static void sort_root(int32_t *arr, int32_t n) {
    if (n <= 1)
        return;
    if (g_max_depth <= 0) {
        local_sort_count_swaps(arr, n);
        return;
    }

    int mid = n / 2;
    int32_t n1 = mid;
    int32_t n2 = n - mid;

    uint16_t chosen = 0;
    int serv = make_server_with_fallback(getpid(), &chosen);
    if (serv < 0)
        exit(1);
    publish_listen_port(chosen);

    pid_t c1 = fork();
    if (c1 < 0) {
        perror("fork");
        close(serv);
        exit(1);
    }
    if (c1 == 0) {
        close(serv);
        worker_main();
    }

    int a1 = accept_conn(serv);
    if (a1 < 0) {
        close(serv);
        kill(c1, SIGTERM);
        waitpid(c1, NULL, 0);
        exit(1);
    }

    int32_t dch = 1;
    if (write_full(a1, &dch, sizeof(dch)) < 0 || write_full(a1, &n1, sizeof(n1)) < 0 ||
        write_full(a1, arr, (size_t)n1 * sizeof(int32_t)) < 0) {
        close(a1);
        close(serv);
        kill(c1, SIGTERM);
        waitpid(c1, NULL, 0);
        exit(1);
    }

    pid_t c2 = fork();
    if (c2 < 0) {
        perror("fork");
        close(a1);
        close(serv);
        kill(c1, SIGTERM);
        waitpid(c1, NULL, 0);
        exit(1);
    }
    if (c2 == 0) {
        close(serv);
        close(a1);
        worker_main();
    }

    int a2 = accept_conn(serv);
    close(serv);
    if (a2 < 0) {
        close(a1);
        kill(c1, SIGTERM);
        kill(c2, SIGTERM);
        waitpid(c1, NULL, 0);
        waitpid(c2, NULL, 0);
        exit(1);
    }

    if (write_full(a2, &dch, sizeof(dch)) < 0 || write_full(a2, &n2, sizeof(n2)) < 0 ||
        write_full(a2, arr + n1, (size_t)n2 * sizeof(int32_t)) < 0) {
        close(a1);
        close(a2);
        kill(c1, SIGTERM);
        kill(c2, SIGTERM);
        waitpid(c1, NULL, 0);
        waitpid(c2, NULL, 0);
        exit(1);
    }

    int32_t sw1 = 0, sw2 = 0, nn1 = 0, nn2 = 0;
    int32_t *L = malloc((size_t)n1 * sizeof(int32_t));
    int32_t *R = malloc((size_t)n2 * sizeof(int32_t));
    if (!L || !R) {
        free(L);
        free(R);
        close(a1);
        close(a2);
        kill(c1, SIGTERM);
        kill(c2, SIGTERM);
        waitpid(c1, NULL, 0);
        waitpid(c2, NULL, 0);
        exit(1);
    }

    if (read_full(a1, &sw1, sizeof(sw1)) < 0 || read_full(a1, &nn1, sizeof(nn1)) < 0 || nn1 != n1 ||
        read_full(a1, L, (size_t)n1 * sizeof(int32_t)) < 0 || read_full(a2, &sw2, sizeof(sw2)) < 0 ||
        read_full(a2, &nn2, sizeof(nn2)) < 0 || nn2 != n2 ||
        read_full(a2, R, (size_t)n2 * sizeof(int32_t)) < 0) {
        free(L);
        free(R);
        close(a1);
        close(a2);
        kill(c1, SIGTERM);
        kill(c2, SIGTERM);
        waitpid(c1, NULL, 0);
        waitpid(c2, NULL, 0);
        exit(1);
    }

    close(a1);
    close(a2);
    waitpid(c1, NULL, 0);
    waitpid(c2, NULL, 0);

    merge_halves(arr, L, n1, R, n2);
    free(L);
    free(R);
}

static int check_sorted(const int32_t *a, int n) {
    for (int i = 0; i < n - 1; i++) {
        if (a[i] > a[i + 1])
            return 0;
    }
    return 1;
}

static int compare_int(const void *a, const void *b) {
    int32_t ia = *(const int32_t *)a;
    int32_t ib = *(const int32_t *)b;
    return (ia > ib) - (ia < ib);
}

static int verify_qsort(const int32_t *a, int n) {
    int32_t *cpy = malloc((size_t)n * sizeof(int32_t));
    if (!cpy)
        return 0;
    memcpy(cpy, a, (size_t)n * sizeof(int32_t));
    qsort(cpy, (size_t)n, sizeof(int32_t), compare_int);
    int ok = memcmp(a, cpy, (size_t)n * sizeof(int32_t)) == 0;
    free(cpy);
    return ok;
}

int main(int argc, char *argv[]) {
    int n = 10000;
    g_max_depth = 3;

    unsetenv(ENV_SRV_PORT);

    if (argc >= 2) {
        n = atoi(argv[1]);
        if (n <= 0) {
            fprintf(stderr, "Некорректный размер: %s\n", argv[1]);
            return 1;
        }
    }
    if (argc >= 3) {
        g_max_depth = atoi(argv[2]);
        if (g_max_depth < 0) {
            fprintf(stderr, "Некорректная глубина: %s\n", argv[2]);
            return 1;
        }
    }

    printf("Сортировка на сокетах (127.0.0.1, порт по PID и запасные при EADDRINUSE), n=%d, "
           "max_depth=%d\n",
           n, g_max_depth);

    int32_t *arr = malloc((size_t)n * sizeof(int32_t));
    if (!arr) {
        perror("malloc");
        return 1;
    }
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++)
        arr[i] = (int32_t)(rand() % 1000000);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    sort_root(arr, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    int ok1 = check_sorted(arr, n);
    int ok2 = verify_qsort(arr, n);

    printf("Время: %.4f с\n", sec);
    printf("Проверка порядка: %s\n", ok1 ? "ОК" : "ОШИБКА");
    printf("Сравнение с qsort: %s\n", ok2 ? "ОК" : "ОШИБКА");

    free(arr);
    return (ok1 && ok2) ? 0 : 1;
}
