package com.example.ru.nstu._1lab;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;
import java.util.Arrays;

public class HelloApplication extends Application {
    @Override
    public void start(Stage stage) throws IOException {
        FXMLLoader fxmlLoader = new FXMLLoader(HelloApplication.class.getResource("simulation-view.fxml"));
        Scene scene = new Scene(fxmlLoader.load(), 800, 600);
        stage.setTitle("Симуляция компании");
        stage.setScene(scene);

        // Сохранение конфигурации при закрытии окна
        stage.setOnCloseRequest(event -> {
            SimulationController controller = fxmlLoader.getController();
            controller.exitApplication();
        });

        stage.show();
    }

    public static void main(String[] args) {
        if (args.length > 0 && "server".equalsIgnoreCase(args[0])) {
            SimulationServer.main(Arrays.copyOfRange(args, 1, args.length));
        } else {
            launch(args);
        }
    }
}
