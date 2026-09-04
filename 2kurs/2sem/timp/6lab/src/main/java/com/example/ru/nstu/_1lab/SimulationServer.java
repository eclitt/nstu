package com.example.ru.nstu._1lab;

import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;

/**
 * TCP-сервер для управления симуляциями клиентов.
 */
public class SimulationServer {
    private final int port;
    private final Map<String, ClientHandler> clients = new ConcurrentHashMap<>();
    private final ExecutorService executor = Executors.newCachedThreadPool();

    public SimulationServer(int port) {
        this.port = port;
    }

    public void start() {
        try (ServerSocket serverSocket = new ServerSocket(port)) {
            System.out.println("Сервер запущен на порту " + port);
            while (!Thread.currentThread().isInterrupted()) {
                Socket clientSocket = serverSocket.accept();
                ClientHandler handler = new ClientHandler(clientSocket);
                executor.execute(handler);
            }
        } catch (IOException e) {
            System.err.println("Ошибка сервера: " + e.getMessage());
        } finally {
            executor.shutdownNow();
        }
    }

    /**
     * Обработчик клиента.
     */
    private class ClientHandler implements Runnable {
        private final Socket socket;
        private ObjectOutputStream out;
        private ObjectInputStream in;
        private String clientId;

        public ClientHandler(Socket socket) {
            this.socket = socket;
            this.clientId = socket.getRemoteSocketAddress().toString();
        }

        @Override
        public void run() {
            try {
                out = new ObjectOutputStream(socket.getOutputStream());
                in = new ObjectInputStream(socket.getInputStream());

                // Добавляем клиента и уведомляем всех
                clients.put(clientId, this);
                broadcastClientList();

                while (true) {
                    Object message = in.readObject();
                    if (message instanceof String) {
                        handleStringMessage((String) message);
                    } else if (message instanceof SwapRequest) {
                        handleSwapRequest((SwapRequest) message);
                    }
                }
            } catch (IOException | ClassNotFoundException e) {
                System.out.println("Клиент " + clientId + " отключился.");
            } finally {
                clients.remove(clientId);
                broadcastClientList();
                try {
                    socket.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        private void handleStringMessage(String msg) {
            if (msg.startsWith("ID:")) {
                String oldId = clientId;
                clientId = msg.substring(3);
                clients.remove(oldId);
                clients.put(clientId, this);
                broadcastClientList();
            }
        }

        private void handleSwapRequest(SwapRequest request) {
            if (request.getStatus() == SwapRequest.Status.PENDING) {
                ClientHandler target = clients.get(request.getTargetClientId());
                if (target != null) {
                    target.sendMessage(request);
                }
            } else {
                // Это ответ (Accepted/Rejected), пересылаем инициатору
                ClientHandler source = clients.get(request.getSourceClientId());
                if (source != null) {
                    source.sendMessage(request);
                }
            }
        }

        public void sendMessage(Object msg) {
            try {
                out.writeObject(msg);
                out.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        private void broadcastClientList() {
            List<String> clientList = new ArrayList<>(clients.keySet());
            for (ClientHandler handler : clients.values()) {
                handler.sendMessage(clientList);
            }
        }
    }

    /**
     * Запрос на обмен объектами.
     */
    public static class SwapRequest implements Serializable {
        public enum Status { PENDING, ACCEPTED, REJECTED }

        private final String sourceClientId;
        private final String targetClientId;
        private final List<Employee> employees;
        private final String giveType; 
        private final String getType;
        private Status status = Status.PENDING;

        public SwapRequest(String source, String target, List<Employee> employees, String giveType, String getType) {
            this.sourceClientId = source;
            this.targetClientId = target;
            this.employees = employees;
            this.giveType = giveType;
            this.getType = getType;
        }

        public String getSourceClientId() { return sourceClientId; }
        public String getTargetClientId() { return targetClientId; }
        public List<Employee> getEmployees() { return employees; }
        public String getGiveType() { return giveType; }
        public String getGetType() { return getType; }
        public Status getStatus() { return status; }
        public void setStatus(Status status) { this.status = status; }
    }

    public static void main(String[] args) {
        int port = 8080;
        if (args.length > 0) {
            try {
                port = Integer.parseInt(args[0]);
            } catch (NumberFormatException e) {
                System.out.println("Неверный порт, используем 8080");
            }
        }
        new SimulationServer(port).start();
    }
}
