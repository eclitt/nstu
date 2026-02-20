package com.example.ru.nstu._practice;

import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.shape.Circle;

public class HelloController {
    private int state;
    @FXML
    private Label welcomeText;
    private Circle circle;

    @FXML
    protected void onHelloButtonClick() {
        if (state == 1) {
            welcomeText.setText("Hello");
            state = 0;
        }
        else {
            welcomeText.setText("Goodbye");
            state = 1;
        };
    }

    @FXML
    protected void onSecondButtonClick() {
        if (state == 1) {
            welcomeText.setText("Nicho");
            state = 0;
        }
        else {
            welcomeText.setText("Chota");
            state = 1;
        };
    }

    @FXML
    protected void onCircleButtonClick() {
        if (state == 1) {
            welcomeText.setText("Nicho");
            state = 0;
        }
        else {
            welcomeText.setText("Chota");
            state = 1;
        };
    }
}
