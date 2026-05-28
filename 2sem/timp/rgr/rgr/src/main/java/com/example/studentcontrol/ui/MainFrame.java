package com.example.studentcontrol.ui;

import com.formdev.flatlaf.FlatDarkLaf;
import com.formdev.flatlaf.FlatLightLaf;

import javax.swing.*;
import java.awt.*;

public class MainFrame extends JFrame {
    public MainFrame() {
        setTitle("Контроль успеваемости студентов");
        setSize(1000, 700);
        setLocationRelativeTo(null);
        setDefaultCloseOperation(EXIT_ON_CLOSE);

        // Базовый современный вид: светлый фон и единый шрифт
        UIManager.put("Table.font", new Font("Segoe UI", Font.PLAIN, 13));
        UIManager.put("Label.font", new Font("Segoe UI", Font.PLAIN, 13));
        UIManager.put("Button.font", new Font("Segoe UI", Font.PLAIN, 13));
        UIManager.put("ComboBox.font", new Font("Segoe UI", Font.PLAIN, 13));
        UIManager.put("TextField.font", new Font("Segoe UI", Font.PLAIN, 13));
        getContentPane().setBackground(new Color(245, 247, 250));

        JToolBar toolbar = new JToolBar();
        toolbar.setFloatable(false);
        toolbar.setBorder(BorderFactory.createEmptyBorder(5, 10, 5, 10));
        JButton btnStudents = new JButton("Студенты");
        JButton btnSubjects = new JButton("Предметы");
        JButton btnGrades = new JButton("Оценки");
        JToggleButton themeToggle = new JToggleButton("Тёмная тема");
        for (AbstractButton b : new AbstractButton[]{btnStudents, btnSubjects, btnGrades, themeToggle}) {
            b.setFocusPainted(false);
        }
        toolbar.add(btnStudents);
        toolbar.add(Box.createHorizontalStrut(10));
        toolbar.add(btnSubjects);
        toolbar.add(Box.createHorizontalStrut(10));
        toolbar.add(btnGrades);
        toolbar.add(Box.createHorizontalGlue());
        toolbar.add(themeToggle);
        add(toolbar, BorderLayout.NORTH);

        CardLayout cardLayout = new CardLayout();
        JPanel cardPanel = new JPanel(cardLayout);
        StudentsPanel sp = new StudentsPanel();
        SubjectsPanel sbp = new SubjectsPanel();
        GradesPanel gp = new GradesPanel();
        cardPanel.add(sp, "STUDENTS");
        cardPanel.add(sbp, "SUBJECTS");
        cardPanel.add(gp, "GRADES");
        add(cardPanel, BorderLayout.CENTER);

        btnStudents.addActionListener(e -> {
            cardLayout.show(cardPanel, "STUDENTS");
        });
        btnSubjects.addActionListener(e -> {
            cardLayout.show(cardPanel, "SUBJECTS");
        });
        btnGrades.addActionListener(e -> {
            cardLayout.show(cardPanel, "GRADES");
        });
        themeToggle.addActionListener(e -> {
            boolean dark = themeToggle.isSelected();
            try {
                if (dark) {
                    FlatDarkLaf.setup();
                    themeToggle.setText("Светлая тема");
                } else {
                    FlatLightLaf.setup();
                    themeToggle.setText("Тёмная тема");
                }
                SwingUtilities.updateComponentTreeUI(this);
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this,
                        "Не удалось переключить тему: " + ex.getMessage(),
                        "Ошибка",
                        JOptionPane.ERROR_MESSAGE);
            }
        });

        cardLayout.show(cardPanel, "STUDENTS");
    }
}