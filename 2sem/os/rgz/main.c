/*
 * Учебная «файловая система» в одном бинарном файле (томе).
 * Параметры (выбраны из ТЗ):
 * - физический уровень: кластеры фиксированного размера
 * - структура свободного пространства: FAT (таблица цепочек кластеров)
 * - базовый уровень: индексный файл имеется (фиксированная таблица дескрипторов)
 * - размещение файла: FAT
 * - логический уровень: каталоги (список имён), сортировка по алфавиту при выводе
 *
 * Команды утилиты:
 *   format <volume> <clusters> <cluster_size> <max_inodes>
 *   ls <volume> <path>
 *   mkdir <volume> <path>
 *   rmdir <volume> <path>
 *   touch <volume> <path>
 *   rm <volume> <path>
 *   mv <volume> <old_path> <new_path>
 *   cat <volume> <path>
 *   write <volume> <path> <string> [--append]
 *   import <volume> <host_src> <fs_dst_path>
 *   export <volume> <fs_src_path> <host_dst>
 *
 * Примечание: функции open/close/seek/read/write реализованы как api для работы с
 * «внутренними» файлами тома (fs_open/fs_close/fs_seek/fs_read/fs_write).
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define VFS_MAGIC "VFS1"
#define VFS_VERSION 1

#define VFS_NAME_MAX 48
#define VFS_PATH_MAX 512

typedef int32_t fat_t; /* 0 = free, -1 = end, >0 = next cluster */

typedef enum {
  INODE_FREE = 0,
  INODE_FILE = 1,
  INODE_DIR = 2,
} inode_type_t;

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
  uint32_t parentq; /* inode index */
  uint32_t size;   /* bytes for file; for dir = 0 */
  uint32_t first_cluster; /* 0 means none */
  char name[VFS_NAME_MAX]; /* null-terminated if shorter */
} inode_t;

typedef struct {
  int in_use;
  uint32_t inode_idx;
  uint32_t pos;
  int can_read;
  int can_write;
  int append;

  /* простая буферизация */
  uint8_t *buf;
  uint32_t buf_cluster; /* cluster index of buffer; 0 if none */
  int buf_dirty;
} fd_entry_t;

typedef struct {
  FILE *fp;
  superblock_t sb;
  fat_t *fat;      /* cluster_count */
  inode_t *inodes; /* max_inodes */
} vfs_t;

static fd_entry_t g_fds[64];

static void die(const char *msg) {
  fprintf(stderr, "Ошибка: %s\n", msg);
  exit(1);
}

static void die_perror(const char *msg) {
  fprintf(stderr, "Ошибка: %s: %s\n", msg, strerror(errno));
  exit(1);
}

static uint64_t round_up(uint64_t x, uint64_t a) {
  return (x + a - 1) / a * a;
}

static uint64_t cluster_to_off(const vfs_t *v, uint32_t cluster) {
  return v->sb.data_offset + (uint64_t)(cluster - 1) * v->sb.cluster_size;
}

static void vfs_flush_fat(const vfs_t *v) {
  if (fseeko(v->fp, (off_t)v->sb.fat_offset, SEEK_SET) != 0) die_perror("seek fat");
  if (fwrite(v->fat, sizeof(fat_t), v->sb.cluster_count, v->fp) != v->sb.cluster_count)
    die_perror("write fat");
}

static void vfs_flush_inodes(const vfs_t *v) {
  if (fseeko(v->fp, (off_t)v->sb.inode_offset, SEEK_SET) != 0) die_perror("seek inodes");
  if (fwrite(v->inodes, sizeof(inode_t), v->sb.max_inodes, v->fp) != v->sb.max_inodes)
    die_perror("write inodes");
}

static void vfs_flush_super(const vfs_t *v) {
  if (fseeko(v->fp, 0, SEEK_SET) != 0) die_perror("seek super");
  if (fwrite(&v->sb, sizeof(v->sb), 1, v->fp) != 1) die_perror("write super");
}

static void vfs_sync(const vfs_t *v) {
  vfs_flush_super(v);
  vfs_flush_fat(v);
  vfs_flush_inodes(v);
  fflush(v->fp);
}

static void vfs_close(vfs_t *v) {
  if (!v) return;
  if (v->fp) fclose(v->fp);
  free(v->fat);
  free(v->inodes);
  memset(v, 0, sizeof(*v));
}

static vfs_t vfs_open(const char *path, const char *mode) {
  vfs_t v;
  memset(&v, 0, sizeof(v));
  v.fp = fopen(path, mode);
  if (!v.fp) die_perror("open volume");

  if (fseeko(v.fp, 0, SEEK_SET) != 0) die_perror("seek super");
  if (fread(&v.sb, sizeof(v.sb), 1, v.fp) != 1) die_perror("read super");

  if (memcmp(v.sb.magic, VFS_MAGIC, 4) != 0) die("Неверный magic тома (нужен format)");
  if (v.sb.version != VFS_VERSION) die("Неподдерживаемая версия тома");

  v.fat = (fat_t *)calloc(v.sb.cluster_count, sizeof(fat_t));
  v.inodes = (inode_t *)calloc(v.sb.max_inodes, sizeof(inode_t));
  if (!v.fat || !v.inodes) die("Недостаточно памяти");

  if (fseeko(v.fp, (off_t)v.sb.fat_offset, SEEK_SET) != 0) die_perror("seek fat");
  if (fread(v.fat, sizeof(fat_t), v.sb.cluster_count, v.fp) != v.sb.cluster_count)
    die_perror("read fat");

  if (fseeko(v.fp, (off_t)v.sb.inode_offset, SEEK_SET) != 0) die_perror("seek inodes");
  if (fread(v.inodes, sizeof(inode_t), v.sb.max_inodes, v.fp) != v.sb.max_inodes)
    die_perror("read inodes");

  return v;
}

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

  /* растягиваем файл до нужного размера */
  uint64_t total = sb.data_offset + (uint64_t)clusters * cluster_size;
  if (fseeko(fp, (off_t)(total - 1), SEEK_SET) != 0) die_perror("seek resize");
  uint8_t z = 0;
  if (fwrite(&z, 1, 1, fp) != 1) die_perror("write resize");
  fflush(fp);
  fclose(fp);
}

static int inode_find_child(const vfs_t *v, uint32_t parent, const char *name) {
  for (uint32_t i = 0; i < v->sb.max_inodes; i++) {
    const inode_t *in = &v->inodes[i];
    if (!in->used) continue;
    if (in->parent != parent) continue;
    if (strncmp(in->name, name, VFS_NAME_MAX) == 0) return (int)i;
  }
  return -1;
}

static int inode_alloc(vfs_t *v) {
  for (uint32_t i = 0; i < v->sb.max_inodes; i++) {
    if (!v->inodes[i].used) {
      memset(&v->inodes[i], 0, sizeof(inode_t));
      v->inodes[i].used = 1;
      return (int)i;
    }
  }
  return -1;
}

static int fat_alloc_cluster(vfs_t *v) {
  for (uint32_t i = 1; i <= v->sb.cluster_count; i++) {
    if (v->fat[i - 1] == 0) {
      v->fat[i - 1] = -1;
      /* обнулим содержимое кластера */
      uint8_t *zero = (uint8_t *)calloc(1, v->sb.cluster_size);
      if (!zero) die("Недостаточно памяти");
      if (fseeko(v->fp, (off_t)cluster_to_off(v, i), SEEK_SET) != 0) die_perror("seek cluster");
      if (fwrite(zero, 1, v->sb.cluster_size, v->fp) != v->sb.cluster_size) die_perror("write cluster");
      free(zero);
      return (int)i;
    }
  }
  return -1;
}

static void fat_free_chain(vfs_t *v, uint32_t first) {
  uint32_t cur = first;
  while (cur != 0) {
    fat_t next = v->fat[cur - 1];
    v->fat[cur - 1] = 0;
    if (next == -1) break;
    if (next <= 0) break;
    cur = (uint32_t)next;
  }
}

static uint32_t fat_nth_cluster(const vfs_t *v, uint32_t first, uint32_t n) {
  uint32_t cur = first;
  for (uint32_t i = 0; i < n; i++) {
    if (cur == 0) return 0;
    fat_t next = v->fat[cur - 1];
    if (next == -1) return 0;
    if (next <= 0) return 0;
    cur = (uint32_t)next;
  }
  return cur;
}

static uint32_t fat_last_cluster(const vfs_t *v, uint32_t first) {
  if (first == 0) return 0;
  uint32_t cur = first;
  while (1) {
    fat_t next = v->fat[cur - 1];
    if (next == -1) return cur;
    if (next <= 0) return cur;
    cur = (uint32_t)next;
  }
}

static uint32_t fat_chain_len(const vfs_t *v, uint32_t first) {
  if (first == 0) return 0;
  uint32_t len = 0;
  uint32_t cur = first;
  while (cur != 0) {
    len++;
    fat_t next = v->fat[cur - 1];
    if (next == -1) break;
    if (next <= 0) break;
    cur = (uint32_t)next;
  }
  return len;
}

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

static void split_path(const char *path, char *parent_out, char *name_out) {
  if (!path || path[0] != '/') die("Путь должен быть абсолютным и начинаться с '/'");
  if (strcmp(path, "/") == 0) die("Нельзя использовать '/' как имя");
  if (strlen(path) >= VFS_PATH_MAX) die("Слишком длинный путь");

  char tmp[VFS_PATH_MAX];
  strncpy(tmp, path, sizeof(tmp) - 1);
  tmp[sizeof(tmp) - 1] = 0;

  char *last = strrchr(tmp, '/');
  if (!last) die("Некорректный путь");
  if (last == tmp) {
    strcpy(parent_out, "/");
    strncpy(name_out, last + 1, VFS_NAME_MAX - 1);
    name_out[VFS_NAME_MAX - 1] = 0;
    return;
  }
  *last = 0;
  strncpy(parent_out, tmp, VFS_PATH_MAX - 1);
  parent_out[VFS_PATH_MAX - 1] = 0;
  strncpy(name_out, last + 1, VFS_NAME_MAX - 1);
  name_out[VFS_NAME_MAX - 1] = 0;
}

static int resolve_path(const vfs_t *v, const char *path) {
  if (!path || path[0] != '/') return -1;
  if (strcmp(path, "/") == 0) return (int)v->sb.root_inode;

  char tmp[VFS_PATH_MAX];
  strncpy(tmp, path, sizeof(tmp) - 1);
  tmp[sizeof(tmp) - 1] = 0;

  uint32_t cur = v->sb.root_inode;
  char *save = NULL;
  for (char *tok = strtok_r(tmp, "/", &save); tok; tok = strtok_r(NULL, "/", &save)) {
    int child = inode_find_child(v, cur, tok);
    if (child < 0) return -1;
    cur = (uint32_t)child;
  }
  return (int)cur;
}

static uint32_t resolve_parent_dir(const vfs_t *v, const char *parent_path) {
  int p = resolve_path(v, parent_path);
  if (p < 0) die("Родительский каталог не найден");
  inode_t *pin = (inode_t *)&v->inodes[p];
  if (pin->type != INODE_DIR) die("Родительский путь не является каталогом");
  return (uint32_t)p;
}

static void cmd_mkdir(vfs_t *v, const char *path) {
  char parent[VFS_PATH_MAX], name[VFS_NAME_MAX];
  split_path(path, parent, name);
  if (name[0] == 0) die("Пустое имя каталога");
  if (strlen(name) >= VFS_NAME_MAX) die("Слишком длинное имя");

  uint32_t pidx = resolve_parent_dir(v, parent);
  if (inode_find_child(v, pidx, name) >= 0) die("Уже существует");

  int idx = inode_alloc(v);
  if (idx < 0) die("Нет свободных дескрипторов (inodes)");
  inode_t *in = &v->inodes[idx];
  in->type = INODE_DIR;
  in->parent = pidx;
  in->size = 0;
  in->first_cluster = 0;
  strncpy(in->name, name, sizeof(in->name) - 1);
}

static bool dir_is_empty(const vfs_t *v, uint32_t dir_idx) {
  for (uint32_t i = 0; i < v->sb.max_inodes; i++) {
    const inode_t *in = &v->inodes[i];
    if (!in->used) continue;
    if (in->parent == dir_idx && i != dir_idx) return false;
  }
  return true;
}

static void cmd_rmdir(vfs_t *v, const char *path) {
  int idx = resolve_path(v, path);
  if (idx < 0) die("Каталог не найден");
  if ((uint32_t)idx == v->sb.root_inode) die("Нельзя удалить корневой каталог");
  inode_t *in = &v->inodes[idx];
  if (in->type != INODE_DIR) die("Это не каталог");
  if (!dir_is_empty(v, (uint32_t)idx)) die("Каталог не пуст");
  in->used = 0;
}

static void cmd_touch(vfs_t *v, const char *path) {
  char parent[VFS_PATH_MAX], name[VFS_NAME_MAX];
  split_path(path, parent, name);
  if (name[0] == 0) die("Пустое имя файла");
  uint32_t pidx = resolve_parent_dir(v, parent);
  if (inode_find_child(v, pidx, name) >= 0) die("Уже существует");

  int idx = inode_alloc(v);
  if (idx < 0) die("Нет свободных дескрипторов (inodes)");
  inode_t *in = &v->inodes[idx];
  in->type = INODE_FILE;
  in->parent = pidx;
  in->size = 0;
  in->first_cluster = 0;
  strncpy(in->name, name, sizeof(in->name) - 1);
}

static void cmd_rm(vfs_t *v, const char *path) {
  int idx = resolve_path(v, path);
  if (idx < 0) die("Файл не найден");
  inode_t *in = &v->inodes[idx];
  if (in->type != INODE_FILE) die("Это не файл");
  if (in->first_cluster) fat_free_chain(v, in->first_cluster);
  in->used = 0;
}

static void cmd_mv(vfs_t *v, const char *old_path, const char *new_path) {
  int idx = resolve_path(v, old_path);
  if (idx < 0) die("Источник не найден");
  if (strcmp(new_path, "/") == 0) die("Некорректный путь назначения");

  char parent[VFS_PATH_MAX], name[VFS_NAME_MAX];
  split_path(new_path, parent, name);
  uint32_t pidx = resolve_parent_dir(v, parent);
  if (inode_find_child(v, pidx, name) >= 0) die("Назначение уже существует");

  inode_t *in = &v->inodes[idx];
  in->parent = pidx;
  memset(in->name, 0, sizeof(in->name));
  strncpy(in->name, name, sizeof(in->name) - 1);
}

typedef struct {
  uint32_t idx;
  char name[VFS_NAME_MAX];
  uint8_t type;
  uint32_t size;
} ls_item_t;

static int cmp_ls_item(const void *a, const void *b) {
  const ls_item_t *x = (const ls_item_t *)a;
  const ls_item_t *y = (const ls_item_t *)b;
  return strncmp(x->name, y->name, VFS_NAME_MAX);
}

static void cmd_ls(const vfs_t *v, const char *path) {
  int didx = resolve_path(v, path);
  if (didx < 0) die("Путь не найден");
  const inode_t *dir = &v->inodes[didx];
  if (dir->type != INODE_DIR) die("Это не каталог");

  ls_item_t *items = (ls_item_t *)calloc(v->sb.max_inodes, sizeof(ls_item_t));
  if (!items) die("Недостаточно памяти");
  size_t n = 0;
  for (uint32_t i = 0; i < v->sb.max_inodes; i++) {
    const inode_t *in = &v->inodes[i];
    if (!in->used) continue;
    if (in->parent != (uint32_t)didx) continue;
    if (i == (uint32_t)didx) continue;
    items[n].idx = i;
    items[n].type = in->type;
    items[n].size = in->size;
    strncpy(items[n].name, in->name, VFS_NAME_MAX - 1);
    n++;
  }
  qsort(items, n, sizeof(ls_item_t), cmp_ls_item);

  for (size_t i = 0; i < n; i++) {
    printf("%c %-48s %10u\n", items[i].type == INODE_DIR ? 'd' : 'f', items[i].name, items[i].size);
  }
  free(items);
}

static void fd_flush_buffer(vfs_t *v, fd_entry_t *fd) {
  if (!fd->in_use || !fd->buf || fd->buf_cluster == 0) return;
  if (!fd->buf_dirty) return;
  if (fseeko(v->fp, (off_t)cluster_to_off(v, fd->buf_cluster), SEEK_SET) != 0) die_perror("seek flush buf");
  if (fwrite(fd->buf, 1, v->sb.cluster_size, v->fp) != v->sb.cluster_size) die_perror("write flush buf");
  fd->buf_dirty = 0;
}

static void fd_load_cluster(vfs_t *v, fd_entry_t *fd, uint32_t cluster) {
  if (!fd->buf) {
    fd->buf = (uint8_t *)malloc(v->sb.cluster_size);
    if (!fd->buf) die("Недостаточно памяти");
  }
  if (fd->buf_cluster == cluster) return;
  fd_flush_buffer(v, fd);
  fd->buf_cluster = cluster;
  if (cluster == 0) {
    memset(fd->buf, 0, v->sb.cluster_size);
    return;
  }
  if (fseeko(v->fp, (off_t)cluster_to_off(v, cluster), SEEK_SET) != 0) die_perror("seek load buf");
  if (fread(fd->buf, 1, v->sb.cluster_size, v->fp) != v->sb.cluster_size) die_perror("read load buf");
}

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

static void fs_close(vfs_t *v, int fdnum) {
  if (fdnum < 0 || fdnum >= (int)(sizeof(g_fds) / sizeof(g_fds[0])) || !g_fds[fdnum].in_use)
    die("Некорректный fd");
  fd_entry_t *fd = &g_fds[fdnum];
  fd_flush_buffer(v, fd);
  free(fd->buf);
  memset(fd, 0, sizeof(*fd));
}

static void fs_seek(int fdnum, int32_t offset, int whence) {
  if (fdnum < 0 || fdnum >= (int)(sizeof(g_fds) / sizeof(g_fds[0])) || !g_fds[fdnum].in_use)
    die("Некорректный fd");
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

static size_t fs_read(vfs_t *v, int fdnum, void *out, size_t nbytes) {
  if (fdnum < 0 || fdnum >= (int)(sizeof(g_fds) / sizeof(g_fds[0])) || !g_fds[fdnum].in_use)
    die("Некорректный fd");
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
  if (fdnum < 0 || fdnum >= (int)(sizeof(g_fds) / sizeof(g_fds[0])) || !g_fds[fdnum].in_use)
    die("Некорректный fd");
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

static void cmd_cat(vfs_t *v, const char *path) {
  int fd = fs_open(v, path, "r");
  inode_t *in = &v->inodes[g_fds[fd].inode_idx];
  uint8_t *buf = (uint8_t *)malloc(in->size + 1);
  if (!buf) die("Недостаточно памяти");
  size_t rd = fs_read(v, fd, buf, in->size);
  buf[rd] = 0;
  fwrite(buf, 1, rd, stdout);
  if (rd && buf[rd - 1] != '\n') putchar('\n');
  free(buf);
  fs_close(v, fd);
}

static void cmd_write(vfs_t *v, const char *path, const char *text, bool append) {
  /* Небольшая обработка escape-последовательностей для удобства тестирования:
     \n \t \\ \" */
  size_t len = strlen(text);
  char *tmp = (char *)malloc(len + 1);
  if (!tmp) die("Недостаточно памяти");
  size_t j = 0;
  for (size_t i = 0; i < len; i++) {
    if (text[i] == '\\' && i + 1 < len) {
      char c = text[i + 1];
      if (c == 'n') {
        tmp[j++] = '\n';
        i++;
        continue;
      }
      if (c == 't') {
        tmp[j++] = '\t';
        i++;
        continue;
      }
      if (c == '\\') {
        tmp[j++] = '\\';
        i++;
        continue;
      }
      if (c == '"') {
        tmp[j++] = '"';
        i++;
        continue;
      }
    }
    tmp[j++] = text[i];
  }
  tmp[j] = 0;

  int fd = fs_open(v, path, append ? "wa" : "wtw");
  fs_write(v, fd, tmp, j);
  fs_close(v, fd);
  free(tmp);
}

static void cmd_import(vfs_t *v, const char *host_src, const char *fs_dst) {
  FILE *in = fopen(host_src, "rb");
  if (!in) die_perror("open host src");
  int fd = fs_open(v, fs_dst, "wtw");

  uint8_t tmp[4096];
  while (1) {
    size_t r = fread(tmp, 1, sizeof(tmp), in);
    if (r > 0) fs_write(v, fd, tmp, r);
    if (r < sizeof(tmp)) {
      if (ferror(in)) die_perror("read host src");
      break;
    }
  }
  fs_close(v, fd);
  fclose(in);
}

static void cmd_export(vfs_t *v, const char *fs_src, const char *host_dst) {
  FILE *out = fopen(host_dst, "wb");
  if (!out) die_perror("open host dst");
  int fd = fs_open(v, fs_src, "r");
  uint8_t tmp[4096];
  while (1) {
    size_t r = fs_read(v, fd, tmp, sizeof(tmp));
    if (r == 0) break;
    if (fwrite(tmp, 1, r, out) != r) die_perror("write host dst");
  }
  fs_close(v, fd);
  fclose(out);
}

static void usage(void) {
  fprintf(stderr,
          "Использование:\n"
          "  vfs format <volume> <clusters> <cluster_size> <max_inodes>\n"
          "  vfs shell <volume>\n"
          "  vfs ls <volume> <path>\n"
          "  vfs mkdir <volume> <path>\n"
          "  vfs rmdir <volume> <path>\n"
          "  vfs touch <volume> <path>\n"
          "  vfs rm <volume> <path>\n"
          "  vfs mv <volume> <old_path> <new_path>\n"
          "  vfs cat <volume> <path>\n"
          "  vfs write <volume> <path> <string> [--append]\n"
          "  vfs import <volume> <host_src> <fs_dst_path>\n"
          "  vfs export <volume> <fs_src_path> <host_dst>\n");
}

static char *trim_left(char *s) {
  while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') s++;
  return s;
}

static void trim_right(char *s) {
  size_t n = strlen(s);
  while (n > 0) {
    char c = s[n - 1];
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      s[n - 1] = 0;
      n--;
      continue;
    }
    break;
  }
}

static void free_argv(char **argv, int argc) {
  if (!argv) return;
  for (int i = 0; i < argc; i++) free(argv[i]);
  free(argv);
}

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

static void shell_help(void) {
  printf("Интерактивный режим (все пути абсолютные):\n");
  printf("  help\n");
  printf("  ls <path>\n");
  printf("  mkdir <path>\n");
  printf("  rmdir <path>\n");
  printf("  touch <path>\n");
  printf("  rm <path>\n");
  printf("  mv <old_path> <new_path>\n");
  printf("  cat <path>\n");
  printf("  write <path> \"строка\" [--append]\n");
  printf("  import <host_src> <fs_dst_path>\n");
  printf("  export <fs_src_path> <host_dst>\n");
  printf("  exit\n");
}

static int run_cmd_on_open_volume(vfs_t *v, int argc, char **argv) {
  if (argc == 0) return 0;
  const char *cmd = argv[0];

  if (strcmp(cmd, "help") == 0) {
    shell_help();
    return 0;
  }
  if (strcmp(cmd, "exit") == 0 || strcmp(cmd, "quit") == 0) return 1;

  if (strcmp(cmd, "ls") == 0) {
    if (argc != 2) die("ls: нужен 1 аргумент <path>");
    cmd_ls(v, argv[1]);
    return 0;
  }
  if (strcmp(cmd, "mkdir") == 0) {
    if (argc != 2) die("mkdir: нужен 1 аргумент <path>");
    cmd_mkdir(v, argv[1]);
    vfs_sync(v);
    return 0;
  }
  if (strcmp(cmd, "rmdir") == 0) {
    if (argc != 2) die("rmdir: нужен 1 аргумент <path>");
    cmd_rmdir(v, argv[1]);
    vfs_sync(v);
    return 0;
  }
  if (strcmp(cmd, "touch") == 0) {
    if (argc != 2) die("touch: нужен 1 аргумент <path>");
    cmd_touch(v, argv[1]);
    vfs_sync(v);
    return 0;
  }
  if (strcmp(cmd, "rm") == 0) {
    if (argc != 2) die("rm: нужен 1 аргумент <path>");
    cmd_rm(v, argv[1]);
    vfs_sync(v);
    return 0;
  }
  if (strcmp(cmd, "mv") == 0) {
    if (argc != 3) die("mv: нужно 2 аргумента <old_path> <new_path>");
    cmd_mv(v, argv[1], argv[2]);
    vfs_sync(v);
    return 0;
  }
  if (strcmp(cmd, "cat") == 0) {
    if (argc != 2) die("cat: нужен 1 аргумент <path>");
    cmd_cat(v, argv[1]);
    return 0;
  }
  if (strcmp(cmd, "write") == 0) {
    if (argc < 3) die("write: нужно <path> <string> [--append]");
    bool append = (argc >= 4 && strcmp(argv[3], "--append") == 0);
    cmd_write(v, argv[1], argv[2], append);
    vfs_sync(v);
    return 0;
  }
  if (strcmp(cmd, "import") == 0) {
    if (argc != 3) die("import: нужно <host_src> <fs_dst_path>");
    cmd_import(v, argv[1], argv[2]);
    vfs_sync(v);
    return 0;
  }
  if (strcmp(cmd, "export") == 0) {
    if (argc != 3) die("export: нужно <fs_src_path> <host_dst>");
    cmd_export(v, argv[1], argv[2]);
    return 0;
  }

  die("Неизвестная команда (help для списка)");
  return 0;
}

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

    int exit_shell = 0;
    /* перехват ошибок через stderr/exit нам подходит для лабораторной;
       но в shell лучше не падать от любой опечатки — обернём простым подходом:
       локальные команды проверяются, а критические ошибки всё равно остановят программу. */
    exit_shell = run_cmd_on_open_volume(&v, argc, argv);
    free_argv(argv, argc);
    if (exit_shell) break;
  }

  free(line);
  vfs_close(&v);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    usage();
    return 1;
  }

  const char *cmd = argv[1];

  if (strcmp(cmd, "format") == 0) {
    if (argc != 6) {
      usage();
      return 1;
    }
    const char *vol = argv[2];
    uint32_t clusters = (uint32_t)strtoul(argv[3], NULL, 10);
    uint32_t clsz = (uint32_t)strtoul(argv[4], NULL, 10);
    uint32_t max_in = (uint32_t)strtoul(argv[5], NULL, 10);
    vfs_format(vol, clusters, clsz, max_in);
    return 0;
  }

  if (strcmp(cmd, "shell") == 0) {
    if (argc != 3) {
      usage();
      return 1;
    }
    cmd_shell(argv[2]);
    return 0;
  }

  if (argc < 4) {
    usage();
    return 1;
  }
  const char *vol = argv[2];
  const char *path = argv[3];

  vfs_t v = vfs_open(vol, "rb+");

  if (strcmp(cmd, "ls") == 0) {
    cmd_ls(&v, path);
  } else if (strcmp(cmd, "mkdir") == 0) {
    cmd_mkdir(&v, path);
    vfs_sync(&v);
  } else if (strcmp(cmd, "rmdir") == 0) {
    cmd_rmdir(&v, path);
    vfs_sync(&v);
  } else if (strcmp(cmd, "touch") == 0) {
    cmd_touch(&v, path);
    vfs_sync(&v);
  } else if (strcmp(cmd, "rm") == 0) {
    cmd_rm(&v, path);
    vfs_sync(&v);
  } else if (strcmp(cmd, "mv") == 0) {
    if (argc != 5) {
      usage();
      vfs_close(&v);
      return 1;
    }
    cmd_mv(&v, argv[3], argv[4]);
    vfs_sync(&v);
  } else if (strcmp(cmd, "cat") == 0) {
    cmd_cat(&v, path);
  } else if (strcmp(cmd, "write") == 0) {
    if (argc < 5) {
      usage();
      vfs_close(&v);
      return 1;
    }
    bool append = false;
    if (argc >= 6 && strcmp(argv[5], "--append") == 0) append = true;
    cmd_write(&v, argv[3], argv[4], append);
    vfs_sync(&v);
  } else if (strcmp(cmd, "import") == 0) {
    if (argc != 5) {
      usage();
      vfs_close(&v);
      return 1;
    }
    cmd_import(&v, argv[3], argv[4]);
    vfs_sync(&v);
  } else if (strcmp(cmd, "export") == 0) {
    if (argc != 5) {
      usage();
      vfs_close(&v);
      return 1;
    }
    cmd_export(&v, argv[3], argv[4]);
  } else {
    usage();
    vfs_close(&v);
    return 1;
  }

  vfs_close(&v);
  return 0;
}
