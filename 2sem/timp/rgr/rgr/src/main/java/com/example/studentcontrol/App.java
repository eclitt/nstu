package com.example.studentcontrol;

import com.example.studentcontrol.dao.DBConnection;
import com.example.studentcontrol.ui.LoginDialog;
import com.example.studentcontrol.ui.MainFrame;
import com.formdev.flatlaf.FlatLightLaf;

import javax.swing.*;
import java.io.IOException;
import java.sql.Connection;
import java.sql.SQLException;

public class App {
    public static void main(String[] args) {
        try {
            FlatLightLaf.setup();
            UIManager.put("Component.arc", 12);
            UIManager.put("Button.arc", 12);
            UIManager.put("TextComponent.arc", 10);
            UIManager.put("Table.showHorizontalLines", true);
            UIManager.put("Table.showVerticalLines", false);
        } catch (Exception e) {
            JOptionPane.showMessageDialog(null,
                    "Не удалось применить тему интерфейса: " + e.getMessage(),
                    "Ошибка", JOptionPane.ERROR_MESSAGE);
            System.exit(1);
        }

        boolean authenticated = LoginDialog.showLoginDialog();
        if (!authenticated) {
            System.exit(0);
        }

        try {
            DBConnection.initializeDatabase();
            try (Connection conn = DBConnection.getConnection()) {
                // connection check
            }
        } catch (SQLException | IOException ex) {
            JOptionPane.showMessageDialog(null,
                    "Не удалось подключиться к БД: " + ex.getMessage(),
                    "Ошибка", JOptionPane.ERROR_MESSAGE);
            System.exit(1);
        }

        SwingUtilities.invokeLater(() -> {
            MainFrame frame = new MainFrame();
            frame.setVisible(true);
        });
    }
}