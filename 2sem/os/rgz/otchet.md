# ВВЕДЕНИЕ

В рамках работы реализована учебная система управления файлами (виртуальная файловая система, ВФС), размещаемая в одном двоичном файле произвольного доступа. ВФС поддерживает базовые операции с файлами и каталогами и демонстрирует принципы организации носителя на уровнях: физическом (кластеры), базовом (индексная таблица), размещения файлов (FAT), логическом (каталоги и пути), а также прикладном (утилита управления).

Дополнительно реализован **интерактивный консольный интерфейс** (режим оболочки), позволяющий работать с томом в диалоговом режиме.

---

# ЦЕЛЬ РАБОТЫ

Разработать и реализовать систему управления файлами в виде «тома» (одного бинарного файла произвольного доступа) с выбранными параметрами из ТЗ:

- разметка носителя кластерами фиксированного размера;
- структура свободного пространства — FAT;
- наличие индексного файла (таблица дескрипторов);
- размещение файлов — FAT-цепочки кластеров;
- поддержка путей, каталогов и базовых операций `open/close/seek/read/write`, а также команд управления носителем.

---

# ТЕОРЕТИЧЕСКИЕ СВЕДЕНИЯ

## Двоичный файл произвольного доступа как «том»

Файл-том хранит метаданные и данные в фиксированных областях (смещениях). Доступ к произвольным позициям реализуется через `fseek/fseeko` и `fread/fwrite`, что позволяет читать и записывать отдельные блоки (кластеры), таблицы и структуры без последовательного прохода по всему файлу.

## Кластеры

Кластер — минимальная единица выделения пространства под содержимое файлов. Данные файла занимают целое число кластеров, а доступ к байтовым смещениям сводится к вычислению:

- номера кластера в цепочке файла: \(k = \lfloor offset / cluster\_size \rfloor\)
- смещения внутри кластера: \(p = offset \bmod cluster\_size\)

## FAT (File Allocation Table)

FAT — таблица, где для каждого кластера хранится:

- `0` — кластер свободен;
- `-1` — конец цепочки (EOC);
- `>0` — номер следующего кластера в цепочке.

Файл представлен как связная цепочка кластеров, начиная с `first_cluster` в дескрипторе.

## Индексный файл (таблица дескрипторов / inode-таблица)

Для упрощения реализации используется фиксированная таблица дескрипторов (`inode`), где каждая запись содержит:

- признак занятости;
- тип (файл/каталог);
- индекс родительского каталога;
- размер файла в байтах;
- первый кластер данных;
- имя объекта.

Каталоги не хранят отдельные «entries» в данных; их содержимое вычисляется сканированием inode-таблицы по полю `parent`.

---

# ОПИСАНИЕ АЛГОРИТМА

## Форматирование тома

1. Создать файл-том и записать `superblock`.
2. Инициализировать FAT нулями (все кластеры свободны).
3. Инициализировать таблицу inode пустыми записями.
4. Создать корневой каталог (`/`) как inode с индексом `0`.
5. «Растянуть» файл-том до размера: `data_offset + cluster_count * cluster_size`.

## Разрешение пути

1. Путь должен быть абсолютным и начинаться с `/`.
2. Стартовая точка — inode корня.
3. По каждому компоненту пути (`strtok` по `/`) выполняется поиск дочернего inode с заданным именем и `parent=current`.
4. Если компонент не найден — ошибка «не существует».

## Создание файла/каталога

1. Разбить путь на `parent_path` и `name`.
2. Разрешить `parent_path` и убедиться, что это каталог.
3. Проверить отсутствие объекта с таким `name` в этом каталоге.
4. Найти свободный inode, заполнить поля (`type`, `parent`, `name`).

## Чтение/запись файла (open/seek/read/write)

- **open**: найти inode по пути (или создать при `w`), создать запись в таблице открытых файлов, при `append` установить `pos=size`, при `truncate` очистить цепочку FAT.
- **seek**: изменить позицию `pos` внутри файла.
- **read**: по текущей позиции вычислить нужный кластер в цепочке и копировать данные порциями до заполнения буфера или EOF.
- **write**: при необходимости расширять цепочку FAT, выделяя новые кластеры; записывать данные порциями, обновляя `size`.
- **буферизация**: используется один кластерный буфер на открытый файл; при смене кластера выполняется flush «грязного» буфера.

## Удаление

- для файла: освободить его FAT-цепочку и пометить inode как свободный;
- для каталога: разрешено удалять только пустой каталог (проверка отсутствия дочерних inode).

---

# ПРОГРАММНАЯ РЕАЛИЗАЦИЯ

Реализация выполнена на языке C в файле `main.c`.

## Как пользоваться утилитой (CLI и shell)

### Сборка

```bash
gcc -O2 -Wall -Wextra -std=c11 -o vfs main.c
```

### Форматирование тома

```bash
./vfs format volume.bin <clusters> <cluster_size> <max_inodes>
```

Пример:

```bash
./vfs format volume.bin 128 1024 128
```

### Режим CLI (каждый раз указывать том)

Формат: `./vfs <команда> <volume> <path> ...`

```bash
./vfs mkdir volume.bin /docs
./vfs touch volume.bin /docs/hello.txt
./vfs write volume.bin /docs/hello.txt "Привет, ВФС!\\n"
./vfs ls volume.bin /docs
./vfs cat volume.bin /docs/hello.txt
./vfs mv volume.bin /docs/hello.txt /docs/hi.txt
./vfs rm volume.bin /docs/hi.txt
./vfs rmdir volume.bin /docs
```

### Режим shell (интерактивно)

```bash
./vfs shell volume.bin
```

Пример:

```bash
vfs> help
vfs> mkdir /docs2
vfs> write /docs2/a.txt "Hi\\n"
vfs> cat /docs2/a.txt
Hi
vfs> ls /docs2
f a.txt                                                      3
vfs> exit
```

## Структуры данных на диске

Ниже приведены основные структуры, которые записываются в файл-том. `superblock_t` хранится в начале тома и содержит размеры/смещения областей, а `inode_t` — элемент индексной таблицы (дескриптор файла/каталога). FAT хранится как массив `fat_t` длиной `cluster_count`.

**Листинг 1 — суперблок и дескриптор (inode).**

```c
typedef int32_t fat_t; /* 0 = free, -1 = end, >0 = next cluster */

typedef struct __attribute__((packed)) {
  char magic[4];
  uint32_t version;
  uint32_t cluster_size;
  uint32_t cluster_count;
  uint32_t max_inodes;

  uint64_t fat_offset;
  uint64_t inode_offset;
  uint64_t data_offset;

  uint32_t root_inode; /* index */
  uint32_t reserved;
} superblock_t;

typedef struct __attribute__((packed)) {
  uint8_t used; /* 0/1 */
  uint8_t type; /* inode_type_t */
  uint16_t reserved0;
  uint32_t parent; /* inode index */
  uint32_t size;   /* bytes for file; for dir = 0 */
  uint32_t first_cluster; /* 0 means none */
  char name[VFS_NAME_MAX]; /* null-terminated if shorter */
} inode_t;
```

## Форматирование тома

Форматирование создаёт файл нужного размера и инициализирует три области: `superblock`, FAT и таблицу inode. После этого записывается корневой каталог `/` (inode с индексом 0).

**Листинг 2 — инициализация тома (`vfs_format`).**

```c
static void vfs_format(const char *path, uint32_t clusters, uint32_t cluster_size, uint32_t max_inodes) {
  if (clusters < 16) die("Слишком мало кластеров (минимум 16)");
  if (cluster_size < 256) die("Слишком маленький размер кластера (минимум 256)");
  if (max_inodes < 16) die("Слишком мало inodes (минимум 16)");

  FILE *fp = fopen(path, "wb+");
  if (!fp) die_perror("create volume");

  superblock_t sb;
  memset(&sb, 0, sizeof(sb));
  memcpy(sb.magic, VFS_MAGIC, 4);
  sb.version = VFS_VERSION;
  sb.cluster_size = cluster_size;
  sb.cluster_count = clusters;
  sb.max_inodes = max_inodes;

  uint64_t off = sizeof(superblock_t);
  sb.fat_offset = round_up(off, 8);
  off = sb.fat_offset + (uint64_t)clusters * sizeof(fat_t);
  sb.inode_offset = round_up(off, 8);
  off = sb.inode_offset + (uint64_t)max_inodes * sizeof(inode_t);
  sb.data_offset = round_up(off, 8);
  sb.root_inode = 0;

  /* записываем суперблок */
  if (fseeko(fp, 0, SEEK_SET) != 0) die_perror("seek super");
  if (fwrite(&sb, sizeof(sb), 1, fp) != 1) die_perror("write super");

  /* FAT */
  fat_t zero = 0;
  if (fseeko(fp, (off_t)sb.fat_offset, SEEK_SET) != 0) die_perror("seek fat");
  for (uint32_t i = 0; i < clusters; i++) {
    if (fwrite(&zero, sizeof(zero), 1, fp) != 1) die_perror("write fat init");
  }

  /* inodes */
  inode_t in0;
  memset(&in0, 0, sizeof(in0));
  if (fseeko(fp, (off_t)sb.inode_offset, SEEK_SET) != 0) die_perror("seek inodes");
  for (uint32_t i = 0; i < max_inodes; i++) {
    if (fwrite(&in0, sizeof(in0), 1, fp) != 1) die_perror("write inode init");
  }

  /* создаём root inode */
  inode_t root;
  memset(&root, 0, sizeof(root));
  root.used = 1;
  root.type = INODE_DIR;
  root.parent = 0;
  root.size = 0;
  root.first_cluster = 0;
  strncpy(root.name, "/", sizeof(root.name) - 1);

  if (fseeko(fp, (off_t)sb.inode_offset, SEEK_SET) != 0) die_perror("seek root inode");
  if (fwrite(&root, sizeof(root), 1, fp) != 1) die_perror("write root inode");
}
```

## Запись с расширением FAT-цепочки

При записи в файл по смещению может потребоваться расширить цепочку кластеров. Функция `ensure_cluster_for_offset` гарантирует, что в FAT выделено достаточно кластеров, и возвращает номер кластера, в который попадает заданное смещение.

**Листинг 3 — выделение кластеров и расширение FAT-цепочки.**

```c
static uint32_t ensure_cluster_for_offset(vfs_t *v, inode_t *in, uint32_t offset) {
  uint32_t need_idx = offset / v->sb.cluster_size; /* 0-based index inside file chain */
  if (in->first_cluster == 0) {
    int c = fat_alloc_cluster(v);
    if (c < 0) die("Нет свободных кластеров");
    in->first_cluster = (uint32_t)c;
  }
  uint32_t have_len = fat_chain_len(v, in->first_cluster);
  while (have_len <= need_idx) {
    uint32_t last = fat_last_cluster(v, in->first_cluster);
    int c = fat_alloc_cluster(v);
    if (c < 0) die("Нет свободных кластеров");
    v->fat[last - 1] = (fat_t)c;
    v->fat[c - 1] = -1;
    have_len++;
  }
  uint32_t cluster = in->first_cluster;
  for (uint32_t i = 0; i < need_idx; i++) {
    fat_t next = v->fat[cluster - 1];
    if (next <= 0) die("Повреждена цепочка FAT");
    cluster = (uint32_t)next;
  }
  return cluster;
}
```

## Реализация open/close/seek/read/write с буферизацией

Операции `open/close/seek/read/write` реализованы в виде внутренних функций `fs_*`. Они работают с таблицей открытых файлов `g_fds[]`, которая хранит текущую позицию, режимы и одно-кластерный буфер. Буферизация устроена так: при обращении к кластеру он загружается в память, а при переходе на другой кластер «грязный» буфер сбрасывается обратно в том.

**Листинг 4 — открытие файла и позиционирование (`fs_open`, `fs_seek`).**

```c
static int fs_open(vfs_t *v, const char *path, const char *mode) {
  int idx = resolve_path(v, path);
  if (idx < 0) {
    if (mode && strchr(mode, 'w')) {
      cmd_touch(v, path);
      idx = resolve_path(v, path);
      if (idx < 0) die("Не удалось создать файл");
    } else {
      die("Файл не найден");
    }
  }
  inode_t *in = &v->inodes[idx];
  if (in->type != INODE_FILE) die("Это не файл");

  int slot = -1;
  for (int i = 0; i < (int)(sizeof(g_fds) / sizeof(g_fds[0])); i++) {
    if (!g_fds[i].in_use) {
      slot = i;
      break;
    }
  }
  if (slot < 0) die("Таблица открытых файлов переполнена");

  fd_entry_t *fd = &g_fds[slot];
  memset(fd, 0, sizeof(*fd));
  fd->in_use = 1;
  fd->inode_idx = (uint32_t)idx;
  fd->pos = 0;
  fd->can_read = (mode && strchr(mode, 'r')) ? 1 : 0;
  fd->can_write = (mode && strchr(mode, 'w')) ? 1 : 0;
  fd->append = (mode && strchr(mode, 'a')) ? 1 : 0;
  if (fd->append) fd->pos = in->size;

  if (fd->can_write && mode && strchr(mode, 't')) {
    /* truncate */
    if (in->first_cluster) fat_free_chain(v, in->first_cluster);
    in->first_cluster = 0;
    in->size = 0;
    fd->pos = 0;
  }
  return slot;
}

static void fs_seek(int fdnum, int32_t offset, int whence) {
  fd_entry_t *fd = &g_fds[fdnum];
  uint32_t newpos = 0;
  if (whence == SEEK_SET) newpos = (offset < 0) ? 0 : (uint32_t)offset;
  else if (whence == SEEK_CUR) {
    int64_t p = (int64_t)fd->pos + offset;
    if (p < 0) p = 0;
    newpos = (uint32_t)p;
  } else die("Неподдерживаемый whence");
  fd->pos = newpos;
}
```

**Листинг 5 — чтение и запись по FAT-цепочке (`fs_read`, `fs_write`).**

```c
static size_t fs_read(vfs_t *v, int fdnum, void *out, size_t nbytes) {
  fd_entry_t *fd = &g_fds[fdnum];
  if (!fd->can_read) die("Файл открыт без чтения");
  inode_t *in = &v->inodes[fd->inode_idx];
  if (fd->pos >= in->size) return 0;

  uint32_t can = in->size - fd->pos;
  if (nbytes > can) nbytes = can;

  uint8_t *dst = (uint8_t *)out;
  size_t done = 0;
  while (done < nbytes) {
    uint32_t off = fd->pos;
    uint32_t cl_index = off / v->sb.cluster_size;
    uint32_t cl_off = off % v->sb.cluster_size;
    uint32_t want = (uint32_t)(nbytes - done);
    uint32_t chunk = v->sb.cluster_size - cl_off;
    if (chunk > want) chunk = want;

    uint32_t cluster = fat_nth_cluster(v, in->first_cluster, cl_index);
    if (cluster == 0) break;
    fd_load_cluster(v, fd, cluster);
    memcpy(dst + done, fd->buf + cl_off, chunk);

    fd->pos += chunk;
    done += chunk;
  }
  return done;
}

static size_t fs_write(vfs_t *v, int fdnum, const void *buf, size_t nbytes) {
  fd_entry_t *fd = &g_fds[fdnum];
  if (!fd->can_write) die("Файл открыт без записи");
  inode_t *in = &v->inodes[fd->inode_idx];
  if (fd->append) fd->pos = in->size;

  const uint8_t *src = (const uint8_t *)buf;
  size_t done = 0;
  while (done < nbytes) {
    uint32_t off = fd->pos;
    uint32_t cl_off = off % v->sb.cluster_size;
    uint32_t want = (uint32_t)(nbytes - done);
    uint32_t chunk = v->sb.cluster_size - cl_off;
    if (chunk > want) chunk = want;

    uint32_t cluster = ensure_cluster_for_offset(v, in, off);
    fd_load_cluster(v, fd, cluster);
    memcpy(fd->buf + cl_off, src + done, chunk);
    fd->buf_dirty = 1;

    fd->pos += chunk;
    done += chunk;
    if (fd->pos > in->size) in->size = fd->pos;
  }
  return done;
}
```

## Интерактивный интерфейс: разбор командной строки и цикл shell

Интерактивный режим реализован командой `./vfs shell <volume>`. Том открывается один раз, далее пользователь вводит команды в цикле. Ввод разбирается функцией `parse_line`, поддерживающей кавычки `"..."` и escape-последовательности (`\\n`, `\\t`, `\\\\`, `\\"`), после чего команда выполняется над уже открытым томом.

**Листинг 6 — разбор строки команды (`parse_line`).**

```c
/* Разбор строки на токены с поддержкой кавычек "..." и escape \n \t \\ \" */
static int parse_line(const char *line, char ***argv_out) {
  *argv_out = NULL;
  int argc = 0;
  int cap = 8;
  char **argv = (char **)calloc((size_t)cap, sizeof(char *));
  if (!argv) die("Недостаточно памяти");

  size_t i = 0;
  while (line[i]) {
    while (line[i] == ' ' || line[i] == '\t') i++;
    if (!line[i]) break;

    int in_quotes = 0;
    if (line[i] == '"') {
      in_quotes = 1;
      i++;
    }

    size_t out_cap = 64;
    size_t out_len = 0;
    char *tok = (char *)malloc(out_cap);
    if (!tok) die("Недостаточно памяти");

    while (line[i]) {
      char c = line[i];
      if (!in_quotes && (c == ' ' || c == '\t')) break;
      if (in_quotes && c == '"') {
        i++;
        break;
      }
      if (c == '\\' && line[i + 1]) {
        char n = line[i + 1];
        if (n == 'n') c = '\n';
        else if (n == 't') c = '\t';
        else if (n == '\\') c = '\\';
        else if (n == '"') c = '"';
        else c = n;
        i += 2;
      } else {
        i++;
      }
      if (out_len + 2 > out_cap) {
        out_cap *= 2;
        char *nt = (char *)realloc(tok, out_cap);
        if (!nt) die("Недостаточно памяти");
        tok = nt;
      }
      tok[out_len++] = c;
    }
    tok[out_len] = 0;

    if (argc + 1 > cap) {
      cap *= 2;
      char **na = (char **)realloc(argv, (size_t)cap * sizeof(char *));
      if (!na) die("Недостаточно памяти");
      argv = na;
    }
    argv[argc++] = tok;
  }

  *argv_out = argv;
  return argc;
}
```

**Листинг 7 — главный цикл оболочки (`cmd_shell`).**

```c
static void cmd_shell(const char *volume_path) {
  vfs_t v = vfs_open(volume_path, "rb+");
  printf("VFS shell: том '%s' открыт. help — список команд.\n", volume_path);

  char *line = NULL;
  size_t cap = 0;
  while (1) {
    printf("vfs> ");
    fflush(stdout);
    ssize_t n = getline(&line, &cap, stdin);
    if (n < 0) break;
    trim_right(line);
    char *s = trim_left(line);
    if (*s == 0) continue;

    char **argv = NULL;
    int argc = parse_line(s, &argv);
    if (argc == 0) {
      free_argv(argv, argc);
      continue;
    }

    int exit_shell = run_cmd_on_open_volume(&v, argc, argv);
    free_argv(argv, argc);
    if (exit_shell) break;
  }

  free(line);
  vfs_close(&v);
}
```

---

# ТЕСТИРОВАНИЕ

Ниже приведён пример сценария ручного тестирования (Linux):

1) Компиляция:

```bash
gcc -O2 -Wall -Wextra -std=c11 -o vfs main.c
```

2) Форматирование тома:

```bash
./vfs format volume.bin 128 1024 128
```

3) Работа с каталогами и файлами:

```bash
./vfs mkdir volume.bin /docs
./vfs touch volume.bin /docs/hello.txt
./vfs write volume.bin /docs/hello.txt "Привет, ВФС!\\n"
./vfs ls volume.bin /docs
./vfs cat volume.bin /docs/hello.txt
```

Ожидаемый результат:
- в `/docs` отображается файл `hello.txt` (тип `f`) с ненулевым размером;
- команда `cat` выводит записанную строку.

4) Импорт/экспорт:

```bash
echo "host file" > host.txt
./vfs import volume.bin host.txt /docs/host.txt
./vfs export volume.bin /docs/host.txt host_out.txt
diff host.txt host_out.txt
```

Ожидаемый результат: файлы идентичны (`diff` без вывода).

## Проверка интерактивного режима (shell)

Запуск оболочки:

```bash
./vfs shell volume.bin
```

Пример сессии:

```bash
vfs> mkdir /docs2
vfs> write /docs2/a.txt "Hi\\n"
vfs> cat /docs2/a.txt
Hi
vfs> ls /docs2
f a.txt                                                      3
vfs> exit
```

---

# ЗАКЛЮЧЕНИЕ

Реализована учебная виртуальная файловая система в одном бинарном файле произвольного доступа. ВФС использует кластеры фиксированного размера, таблицу FAT для управления свободным пространством и размещением файлов, а также индексную таблицу дескрипторов для метаданных. Реализованы операции с каталогами и файлами и утилита управления томом.

---

# ПРИЛОЖЕНИЕ

Файл исходного кода: `2sem/os/rgz/main.c`.

