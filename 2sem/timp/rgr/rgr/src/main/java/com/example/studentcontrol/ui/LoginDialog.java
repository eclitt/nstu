package com.example.studentcontrol.ui;

import javax.swing.*;
import java.awt.*;

public class LoginDialog extends JDialog {
    private boolean authenticated;
    private final JTextField tfLogin = new JTextField(16);
    private final JPasswordField pfPassword = new JPasswordField(16);

    private LoginDialog() {
        setTitle("Вход в аккаунт");
        setModal(true);
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        setLayout(new BorderLayout(10, 10));
        setResizable(false);

        JPanel form = new JPanel(new GridBagLayout());
        form.setBorder(BorderFactory.createEmptyBorder(12, 12, 4, 12));
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(6, 6, 6, 6);
        gbc.fill = GridBagConstraints.HORIZONTAL;

        gbc.gridx = 0;
        gbc.gridy = 0;
        form.add(new JLabel("Логин:"), gbc);
        gbc.gridx = 1;
        form.add(tfLogin, gbc);

        gbc.gridx = 0;
        gbc.gridy = 1;
        form.add(new JLabel("Пароль:"), gbc);
        gbc.gridx = 1;
        form.add(pfPassword, gbc);

        JButton btnLogin = new JButton("Войти");
        JButton btnCancel = new JButton("Отмена");

        JPanel controls = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 8));
        controls.add(btnCancel);
        controls.add(btnLogin);

        btnLogin.addActionListener(e -> tryLogin());
        btnCancel.addActionListener(e -> dispose());
        getRootPane().setDefaultButton(btnLogin);

        add(form, BorderLayout.CENTER);
        add(controls, BorderLayout.SOUTH);
        pack();
        setLocationRelativeTo(null);
    }

    private void tryLogin() {
        String login = tfLogin.getText().trim();
        String password = new String(pfPassword.getPassword());

        // Учебная авторизация, можно заменить на БД/LDAP
        if ("admin".equals(login) && "1234".equals(password)) {
            authenticated = true;
            dispose();
            return;
        }
        JOptionPane.showMessageDialog(this,
                "Неверный логин или пароль",
                "Ошибка авторизации",
                JOptionPane.ERROR_MESSAGE);
        pfPassword.setText("");
        pfPassword.requestFocusInWindow();
    }

    public static boolean showLoginDialog() {
        LoginDialog dialog = new LoginDialog();
        dialog.setVisible(true);
        return dialog.authenticated;
    }
}
