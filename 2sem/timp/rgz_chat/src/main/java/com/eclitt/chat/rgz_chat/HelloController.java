package com.eclitt.chat.rgz_chat;

import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.input.MouseEvent;

public class HelloController {
    @FXML
    private TextField messageField;
    @FXML
    private Button sendButton;
    @FXML
    private TextArea chatTextField;

    @FXML
    private void sendMessage() {
            chatTextField.setText(messageField.getText());
            messageField.clear();
            sendButton.setText("Sended");
    }

}