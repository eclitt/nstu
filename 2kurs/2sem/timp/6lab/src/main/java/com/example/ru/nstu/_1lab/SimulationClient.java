package com.example.ru.nstu._1lab;

import javafx.application.Platform;
import java.io.*;
import java.net.*;
import java.util.*;
import java.util.function.Consumer;

/**
 * Сетевой клиент для взаимодействия с сервером симуляции.
 */
public class SimulationClient {
    private Socket socket;
    private ObjectOutputStream out;
    private ObjectInputStream in;
    private String serverHost;
    private int serverPort;
    private String myId;
    private boolean connected = false;

    private Consumer<List<String>> onClientListUpdated;
    private Consumer<SimulationServer.SwapRequest> onSwapRequestReceived;

    public SimulationClient(String host, int port, String myId) {
        this.serverHost = host;
        this.serverPort = port;
        this.myId = myId;
    }

    public void connect() throws IOException {
        socket = new Socket();
        socket.connect(new InetSocketAddress(serverHost, serverPort), 5000);
        out = new ObjectOutputStream(socket.getOutputStream());
        in = new ObjectInputStream(socket.getInputStream());
        connected = true;

        // Отправляем свой ID серверу
        sendMessage("ID:" + myId);

        // Поток для чтения сообщений
        Thread readerThread = new Thread(this::listen);
        readerThread.setDaemon(true);
        readerThread.start();
    }

    private void listen() {
        try {
            while (connected) {
                Object msg = in.readObject();
                if (msg instanceof List) {
                    List<String> clients = (List<String>) msg;
                    if (onClientListUpdated != null) {
                        Platform.runLater(() -> onClientListUpdated.accept(clients));
                    }
                } else if (msg instanceof SimulationServer.SwapRequest) {
                    SimulationServer.SwapRequest request = (SimulationServer.SwapRequest) msg;
                    if (onSwapRequestReceived != null) {
                        Platform.runLater(() -> onSwapRequestReceived.accept(request));
                    }
                }
            }
        } catch (IOException | ClassNotFoundException e) {
            Platform.runLater(() -> SimulationController.showAlert("Ошибка", "Потерянно соединение с сервером" + e.toString()));
            System.err.println("Связь с сервером потеряна: " + e.toString());
            connected = false;
        }
    }

    public void sendSwapRequest(String targetId, List<Employee> employees, String giveType, String getType) {
        if (!connected) {
            Platform.runLater(() -> SimulationController.showAlert("Ошибка", "Нет соединение с сервером"));
            return;
        }
        SimulationServer.SwapRequest request = new SimulationServer.SwapRequest(myId, targetId, employees, giveType, getType);
        sendMessage(request);
    }

    public void sendSwapResponse(SimulationServer.SwapRequest request) {
        if (!connected) return;
        sendMessage(request);
    }

    private synchronized void sendMessage(Object msg) {
        try {
            out.writeObject(msg);
            out.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void setOnClientListUpdated(Consumer<List<String>> callback) {
        this.onClientListUpdated = callback;
    }

    public void setOnSwapRequestReceived(Consumer<SimulationServer.SwapRequest> callback) {
        this.onSwapRequestReceived = callback;
    }

    public void disconnect() {
        connected = false;
        try {
            if (socket != null) socket.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public boolean isConnected() {
        return connected;
    }
}
