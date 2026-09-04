package com.example.ru.nstu._1lab;

import java.util.List;

/**
 * Абстрактный класс, описывающий «интеллектуальное поведение» объектов.
 * Создаёт поток, обеспечивающий движение объектов определённого типа.
 * Поддерживает остановку/возобновление через wait()/notify() и настройку приоритета потока.
 */
public abstract class BaseAI {

    /** Список объектов, которыми управляет данный AI */
    protected final List<? extends Employee> employees;

    /** Поток, в котором выполняется расчёт движения */
    protected Thread aiThread;

    /** Флаг работы потока */
    protected volatile boolean running = false;

    /** Флаг паузы (засыпами управляет данный ние через wait) */
    protected volatile boolean paused = false;

    /** Объект синхронизации для wait/notify */
    protected final Object pauseLock = new Object();

    /** Скорость движения (пикселей в секунду) */
    protected double velocity = 50.0;

    /** Границы рабочей области */
    protected double worldWidth;
    protected double worldHeight;

    /**
     * Конструктор.
     *
     * @param employees   список объектов для управления
     * @param worldWidth  ширина рабочей области
     * @param worldHeight высота рабочей области
     */
    public BaseAI(List<? extends Employee> employees, double worldWidth, double worldHeight) {
        this.employees = employees;
        this.worldWidth = worldWidth;
        this.worldHeight = worldHeight;
    }

    /**
     * Запускает поток AI. Если поток уже запущен — ничего не делает.
     */
    public synchronized void start() {
        if (running && aiThread != null && aiThread.isAlive()) {
            return;
        }
        running = true;
        paused = false;
        aiThread = new Thread(this::runLoop, getThreadName());
        aiThread.setDaemon(true);
        aiThread.start();
    }

    /**
     * Останавливает поток AI.
     */
    public synchronized void stop() {
        running = false;
        // Пробуждаем поток, если он в wait()
        synchronized (pauseLock) {
            pauseLock.notifyAll();
        }
        if (aiThread != null) {
            try {
                aiThread.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Приостанавливает работу AI (засыпание через wait).
     */
    public void pause() {
        paused = true;
    }

    /**
     * Возобновляет работу AI (notifyAll).
     */
    public void resume() {
        synchronized (pauseLock) {
            paused = false;
            pauseLock.notifyAll();
        }
    }

    /**
     * Возвращает true, если AI запущен.
     */
    public boolean isRunning() {
        return running && aiThread != null && aiThread.isAlive();
    }

    /**
     * Возвращает true, если AI на паузе.
     */
    public boolean isPaused() {
        return paused;
    }

    /**
     * Основной цикл потока.
     */
    protected void runLoop() {
        while (running) {
            // Проверка паузы
            synchronized (pauseLock) {
                while (paused && running) {
                    try {
                        pauseLock.wait();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return;
                    }
                }
            }

            if (!running) break;

            // Обновление позиций всех объектов
            updatePositions();

            // Пауза между обновлениями (~60 FPS)
            try {
                Thread.sleep(16);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    /**
     * Обновляет позиции всех объектов. Вызывается в потоке AI.
     */
    protected abstract void updatePositions();

    /**
     * Возвращает имя потока (для отладки).
     */
    protected abstract String getThreadName();

    /**
     * Устанавливает приоритет потока.
     *
     * @param priority приоритет (1-10, соответствует Thread.MIN_PRIORITY — Thread.MAX_PRIORITY)
     */
    public void setThreadPriority(int priority) {
        if (aiThread != null) {
            aiThread.setPriority(Math.max(Thread.MIN_PRIORITY, Math.min(Thread.MAX_PRIORITY, priority)));
        }
    }

    /**
     * Возвращает текущий приоритет потока.
     */
    public int getThreadPriority() {
        if (aiThread != null) {
            return aiThread.getPriority();
        }
        return Thread.NORM_PRIORITY;
    }

    /**
     * Устанавливает скорость движения.
     */
    public void setVelocity(double velocity) {
        this.velocity = velocity;
    }

    /**
     * Возвращает скорость движения.
     */
    public double getVelocity() {
        return velocity;
    }

    /**
     * Устанавливает границы мира.
     */
    public void setWorldBounds(double width, double height) {
        this.worldWidth = width;
        this.worldHeight = height;
    }
}
