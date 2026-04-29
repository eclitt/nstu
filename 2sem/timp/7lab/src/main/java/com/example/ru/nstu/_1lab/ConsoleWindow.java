package com.example.ru.nstu._1lab;

import javafx.application.Platform;
import javafx.scene.Scene;
import javafx.scene.control.TextArea;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.Pipe;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ConsoleWindow {
    private final Stage stage;
    private final TextArea textArea;
    private final Pipe pipe;
    private final SimulationController controller;

    public ConsoleWindow(SimulationController controller) throws IOException {
        this.controller = controller;
        this.stage = new Stage();
        this.stage.setTitle("Консоль");

        this.textArea = new TextArea();
        this.textArea.setEditable(true);
        this.textArea.setStyle("-fx-font-family: 'Courier New';");
        this.textArea.setText("> ");
        this.textArea.positionCaret(2);
        VBox.setVgrow(textArea, Priority.ALWAYS);

        VBox root = new VBox(textArea);
        Scene scene = new Scene(root, 400, 300);
        this.stage.setScene(scene);

        // Создаем канал для передачи команд
        this.pipe = Pipe.open();

        // Поток для чтения из канала
        Thread commandProcessor = new Thread(this::processCommands);
        commandProcessor.setDaemon(true);
        commandProcessor.start();

        this.textArea.setOnKeyPressed(event -> {
            if (event.getCode() == KeyCode.ENTER) {
                String[] lines = textArea.getText().split("\n");
                if (lines.length > 0) {
                    String lastLine = lines[lines.length - 1].trim();
                    String lastCommand = lastLine;
                    if (lastLine.startsWith(">")) {
                        lastCommand = lastLine.substring(1).trim();
                    }
                    if (!lastCommand.isEmpty()) {
                        sendCommand(lastCommand);
                    }
                }
            }
        });
    }

    public void show() {
        if (!stage.isShowing()) {
            stage.show();
        } else {
            stage.toFront();
        }
    }

    public void close() {
        stage.close();
        try {
            pipe.sink().close();
            pipe.source().close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void sendCommand(String command) {
        try {
            ByteBuffer buffer = ByteBuffer.allocate(1024);
            buffer.clear();
            buffer.put(command.getBytes(StandardCharsets.UTF_8));
            buffer.flip();
            while (buffer.hasRemaining()) {
                pipe.sink().write(buffer);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void processCommands() {
        ByteBuffer buffer = ByteBuffer.allocate(1024);
        try {
            while (true) {
                buffer.clear();
                int bytesRead = pipe.source().read(buffer);
                if (bytesRead > 0) {
                    buffer.flip();
                    byte[] bytes = new byte[buffer.remaining()];
                    buffer.get(bytes);
                    String command = new String(bytes, StandardCharsets.UTF_8);
                    handleCommand(command);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void handleCommand(String command) {
        Platform.runLater(() -> {
            String response;
            String trimmedCommand = command.trim();
            
            // Паттерн для команды "Нанять N новых менеджеров"
            Pattern hirePattern = Pattern.compile("(?i)Нанять\\s+(\\d+)\\s+новых\\s+менеджеров");
            Matcher hireMatcher = hirePattern.matcher(trimmedCommand);

            if (trimmedCommand.equalsIgnoreCase("Уволить всех менеджеров")) {
                int count = controller.fireAllManagers();
                response = "\nСистема: Уволено " + count + " менеджеров.";
            } else if (hireMatcher.matches()) {
                try {
                    int n = Integer.parseInt(hireMatcher.group(1));
                    if (n>=999) {
                        response = "\nСистема: Нельзя нанять больше 1000 новых менеджеров.";
                    } else {
                        controller.hireManagers(n);
                        response = "\nСистема: Нанято " + n + " новых менеджеров.";
                    }
                } catch (NumberFormatException e) {
                    response = "\nСистема: Ошибка в параметре N.";
                }
            } else {
                response = "\nСистема: Неизвестная команда.";
            }
            textArea.appendText(response + "\n> ");
        });
    }
}
