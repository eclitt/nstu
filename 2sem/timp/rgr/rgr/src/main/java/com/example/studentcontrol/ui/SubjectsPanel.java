package com.example.studentcontrol.ui;

import com.example.studentcontrol.dao.SubjectDAO;
import com.example.studentcontrol.model.Subject;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.sql.SQLException;
import java.util.List;

public class SubjectsPanel extends JPanel {
    private JTable table;
    private DefaultTableModel model;
    private JTextField tfName;
    private JSpinner spSemester;
    private SubjectDAO dao = new SubjectDAO();

    public SubjectsPanel() {
        setLayout(new BorderLayout(10, 10));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
        model = new DefaultTableModel(new String[]{"ID","Название","Семестр"}, 0);
        table = new JTable(model);
        table.setFillsViewportHeight(true);
        JScrollPane scrollPane = new JScrollPane(table);
        scrollPane.setBorder(BorderFactory.createTitledBorder("Список предметов"));
        add(scrollPane, BorderLayout.CENTER);

        JPanel form = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5,5,5,5);
        gbc.fill = GridBagConstraints.HORIZONTAL;

        tfName = new JTextField(10);
        spSemester = new JSpinner(new SpinnerNumberModel(1, 1, 8, 1));

        // Row 0 - Name
        gbc.gridx = 0; gbc.gridy = 0;
        form.add(new JLabel("Название:"), gbc);
        gbc.gridx = 1; gbc.gridy = 0;
        form.add(tfName, gbc);

        // Row 0 - Semester
        gbc.gridx = 2; gbc.gridy = 0;
        form.add(new JLabel("Семестр:"), gbc);
        gbc.gridx = 3; gbc.gridy = 0;
        form.add(spSemester, gbc);

        // Row 1 - Buttons
        gbc.gridx = 1; gbc.gridy = 1;
        JButton btnAdd = new JButton("Добавить");
        form.add(btnAdd, gbc);
        gbc.gridx = 2; gbc.gridy = 1;
        JButton btnDelete = new JButton("Удалить");
        form.add(btnDelete, gbc);

        add(form, BorderLayout.SOUTH);

        btnAdd.addActionListener(e -> {
            try {
                dao.insert(new Subject(0, tfName.getText(), (int) spSemester.getValue()));
                refresh();
            } catch (SQLException ex) {
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
            List<Subject> list = dao.findAll();
            for (Subject s : list) {
                model.addRow(new Object[]{s.getId(), s.getName(), s.getSemester()});
            }
        } catch (SQLException ex) {
            JOptionPane.showMessageDialog(this, "Ошибка: " + ex.getMessage());
        }
    }
}
