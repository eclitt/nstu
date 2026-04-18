import random
import statistics
import time

# Параметры
N = 1000
INTERVAL_POST = (1, 11)
INTERVAL_OBS = (1, 19)
COMPUTERS = 2
EXPERIMENTS = 100

def simulate_one():
    # Очередь (храним только время поступления)
    queue = []
    
    # Компьютеры: 0 = свободен, иначе время освобождения
    comp_free = [0.0, 0.0]
    
    # Статистика
    wait_times = []
    system_times = []
    idle_time = [0.0, 0.0]
    
    # Начальные условия
    t_curr = 0.0
    t_next_arrival = random.uniform(*INTERVAL_POST)
    processed = 0
    
    while processed < N:
        # Какое событие раньше?
        t_next_event = t_next_arrival
        comp_idx = -1
        
        for i in range(COMPUTERS):
            if comp_free[i] > t_curr and comp_free[i] < t_next_event:
                t_next_event = comp_free[i]
                comp_idx = i
        
        # Считаем время простоя
        delta = t_next_event - t_curr
        for i in range(COMPUTERS):
            if comp_free[i] <= t_curr:
                idle_time[i] += delta
        
        t_curr = t_next_event
        
        # Обработка события
        if t_curr == t_next_arrival:
            # ПОСТУПЛЕНИЕ
            arrival = t_curr
            
            # Ищем свободный компьютер (сначала первый)
            assigned = False
            for i in range(COMPUTERS):
                if comp_free[i] <= t_curr:
                    obs = random.uniform(*INTERVAL_OBS)
                    comp_free[i] = t_curr + obs
                    system_times.append(obs)
                    wait_times.append(0.0)
                    assigned = True
                    break
            
            if not assigned:
                queue.append(arrival)
            
            t_next_arrival = t_curr + random.uniform(*INTERVAL_POST)
            processed += 1
        
        else:
            # ОСВОБОЖДЕНИЕ компьютера comp_idx
            if queue:
                arrival = queue.pop(0)
                wait = t_curr - arrival
                obs = random.uniform(*INTERVAL_OBS)
                comp_free[comp_idx] = t_curr + obs
                system_times.append(wait + obs)
                wait_times.append(wait)
            else:
                comp_free[comp_idx] = 0.0
    
    total_time = t_curr
    idle_probs = [idle_time[i] / total_time for i in range(COMPUTERS)]
    
    return (
        statistics.mean(wait_times) if wait_times else 0.0,
        statistics.mean(system_times) if system_times else 0.0,
        idle_probs
    )

# --- Запуск ---
print(f"Запуск {EXPERIMENTS} экспериментов...")
start = time.time()

results = []
for exp in range(EXPERIMENTS):
    results.append(simulate_one())
    if (exp + 1) % 10 == 0:
        print(f"✅ {exp + 1}/{EXPERIMENTS}")

# Собираем статистику
wait_avgs = [r[0] for r in results]
system_avgs = [r[1] for r in results]
idle_probs = list(zip(*[r[2] for r in results]))

print("\n" + "="*50)
print("РЕЗУЛЬТАТЫ")
print("="*50)
print(f"\n⏱️ Время выполнения: {time.time() - start:.1f} сек")
print(f"\n📌 Среднее время ожидания в очереди:")
print(f"   {statistics.mean(wait_avgs):.4f} ± {statistics.stdev(wait_avgs):.4f}")
print(f"\n📌 Среднее время в системе:")
print(f"   {statistics.mean(system_avgs):.4f} ± {statistics.stdev(system_avgs):.4f}")
print(f"\n📌 Вероятность простоя:")
for i in range(COMPUTERS):
    print(f"   Комп{i+1}: {statistics.mean(idle_probs[i]):.4f} ± {statistics.stdev(idle_probs[i]):.4f}")