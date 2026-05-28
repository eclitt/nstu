package com.example.studentcontrol.ui;

import com.example.studentcontrol.dao.StudentDAO;
import com.example.studentcontrol.model.Student;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import javax.swing.text.DateFormatter;
import java.awt.*;
import java.sql.SQLException;
import java.text.SimpleDateFormat;
import java.time.LocalDate;
import java.util.List;

public class StudentsPanel extends JPanel {
    private JTable table;
    private DefaultTableModel model;
    private JTextField tfFirst, tfLast, tfGroup;
    private JFormattedTextField tfDate;
    private StudentDAO dao = new StudentDAO();

    public StudentsPanel() {
        setLayout(new BorderLayout(10, 10));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        model = new DefaultTableModel(new String[]{"ID","Фамилия","Имя","Группа","Дата"}, 0);
        table = new JTable(model);
        table.setFillsViewportHeight(true);
        JScrollPane scrollPane = new JScrollPane(table);
        scrollPane.setBorder(BorderFactory.createTitledBorder("Список студентов"));
        add(scrollPane, BorderLayout.CENTER);

        // Form panel with GridBagLayout
        JPanel form = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5,5,5,5);
        gbc.fill = GridBagConstraints.HORIZONTAL;

        tfLast = new JTextField(10);
        tfFirst = new JTextField(10);
        tfGroup = new JTextField(8);

        DateFormatter dateFormatter = new DateFormatter(new SimpleDateFormat("yyyy-MM-dd"));
        tfDate = new JFormattedTextField(dateFormatter);
        tfDate.setColumns(10);

        // Row 0 - Last Name
        gbc.gridx = 0; gbc.gridy = 0;
        form.add(new JLabel("Фамилия:"), gbc);
        gbc.gridx = 1; gbc.gridy = 0;
        form.add(tfLast, gbc);

        // Row 0 - First Name
        gbc.gridx = 2; gbc.gridy = 0;
        form.add(new JLabel("Имя:"), gbc);
        gbc.gridx = 3; gbc.gridy = 0;
        form.add(tfFirst, gbc);

        // Row 1 - Group
        gbc.gridx = 0; gbc.gridy = 1;
        form.add(new JLabel("Группа:"), gbc);
        gbc.gridx = 1; gbc.gridy = 1;
        form.add(tfGroup, gbc);

        // Row 1 - Date
        gbc.gridx = 2; gbc.gridy = 1;
        form.add(new JLabel("Дата (YYYY-MM-DD):"), gbc);
        gbc.gridx = 3; gbc.gridy = 1;
        form.add(tfDate, gbc);

        // Row 2 - Buttons
        gbc.gridx = 1; gbc.gridy = 2;
        JButton btnAdd = new JButton("Добавить");
        form.add(btnAdd, gbc);
        gbc.gridx = 2; gbc.gridy = 2;
        JButton btnDelete = new JButton("Удалить");
        form.add(btnDelete, gbc);

        add(form, BorderLayout.SOUTH);

        // Action listeners
        btnAdd.addActionListener(e -> {
            try {
                LocalDate date = LocalDate.parse(tfDate.getText());
                dao.insert(new Student(0, tfFirst.getText(), tfLast.getText(), tfGroup.getText(), date));
                refresh();
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Ошибка: " + ex.getMessage());
            }
        });
        btnDelete.addActionListener(e -> {
            int row = table.getSelectedRow();
            if (row >= 0) {
                int id = (int) model.getValueAt(row, 0);
                try {
                    dao.delete(id);
                    refresh();
                } catch (SQLException ex) {
                    JOptionPane.showMessageDialog(this, "Ошибка: " + ex.getMessage());
                }
            }
        });

        refresh();
    }

    private void refresh() {
        model.setRowCount(0);
        try {
            List<Student> list = dao.findAll();
            for (Student s : list) {
                model.addRow(new Object[]{
                        s.getId(),
                        s.getLastName(),
                        s.getFirstName(),
                        s.getGroupName(),
                        s.getEnrollmentDate().toString()
                });
            }
        } catch (SQLException ex) {
            JOptionPane.showMessageDialog(this, "Ошибка: " + ex.getMessage());
        }
    }

}
