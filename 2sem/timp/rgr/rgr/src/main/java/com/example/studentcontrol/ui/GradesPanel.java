package com.example.studentcontrol.ui;

import com.example.studentcontrol.dao.GradeDAO;
import com.example.studentcontrol.dao.StudentDAO;
import com.example.studentcontrol.dao.SubjectDAO;
import com.example.studentcontrol.model.Grade;
import com.example.studentcontrol.model.Student;
import com.example.studentcontrol.model.Subject;
import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.sql.SQLException;
import java.util.List;

public class GradesPanel extends JPanel {
    private JComboBox<Student> cbStudents;
    private JComboBox<Subject> cbSubjects;
    private JSpinner spGrade;
    private DefaultTableModel model;
    private JTable table;
    private GradeDAO gradeDao = new GradeDAO();
    private StudentDAO studentDao = new StudentDAO();
    private SubjectDAO subjectDao = new SubjectDAO();

    public GradesPanel() {
        setLayout(new BorderLayout(10, 10));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        JPanel top = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(5,5,5,5);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        cbStudents = new JComboBox<>();
        cbSubjects = new JComboBox<>();
        spGrade = new JSpinner(new SpinnerNumberModel(1,1,10,1));
        JButton btnAdd = new JButton("Добавить оценку");

        gbc.gridx = 0; gbc.gridy = 0;
        top.add(new JLabel("Студент:"), gbc);
        gbc.gridx = 1;
        top.add(cbStudents, gbc);

        gbc.gridx = 2;
        top.add(new JLabel("Предмет:"), gbc);
        gbc.gridx = 3;
        top.add(cbSubjects, gbc);

        gbc.gridx = 0; gbc.gridy = 1;
        top.add(new JLabel("Оценка:"), gbc);
        gbc.gridx = 1;
        top.add(spGrade, gbc);

        gbc.gridx = 3;
        top.add(btnAdd, gbc);
        add(top, BorderLayout.NORTH);

        model = new DefaultTableModel(new String[]{"ID","Предмет","Оценка","Дата"},0);
        table = new JTable(model);
        table.setFillsViewportHeight(true);
        JScrollPane scrollPane = new JScrollPane(table);
        scrollPane.setBorder(BorderFactory.createTitledBorder("Оценки выбранного студента"));
        add(scrollPane, BorderLayout.CENTER);

        cbStudents.addActionListener(e -> refreshGrades());
        btnAdd.addActionListener(e -> {
            Student s = (Student)cbStudents.getSelectedItem();
            Subject sub = (Subject)cbSubjects.getSelectedItem();
            int gr = (int)spGrade.getValue();
            try {
                gradeDao.insert(s.getId(), sub.getId(), gr);
                refreshGrades();
            } catch(SQLException ex) {
                JOptionPane.showMessageDialog(this, "Ошибка: "+ex.getMessage());
            }
        });

        this.addComponentListener(new ComponentAdapter() {
            @Override
            public void componentShown(ComponentEvent e) {
                refreshStudents();
                refreshSubjects();
                refreshGrades();
            }
        });

        // populate combos
        refreshStudents();
        refreshSubjects();
    }

    private void refreshStudents() {
        try {
            cbStudents.removeAllItems();
            List<Student> students = studentDao.findAll();
            for(Student s : students) cbStudents.addItem(s);
        } catch(SQLException ex){
            JOptionPane.showMessageDialog(this, "Ошибка загрузки студентов: "+ex.getMessage());
        }
    }
    private void refreshSubjects() {
        try {
            cbSubjects.removeAllItems();
            List<Subject> list = subjectDao.findAll();
            for(Subject s : list) cbSubjects.addItem(s);
        } catch(SQLException ex){
            JOptionPane.showMessageDialog(this, "Ошибка загрузки предметов: "+ex.getMessage());
        }
    }
    private void refreshGrades() {
        model.setRowCount(0);
        Student s = (Student)cbStudents.getSelectedItem();
        if(s==null) return;
        try {
            List<Grade> list = gradeDao.findByStudent(s.getId());
            for(Grade g : list){
                model.addRow(new Object[]{g.getId(), g.getSubject().getName(), g.getGrade(), g.getDateAssigned()});
            }
        } catch(SQLException ex){
            JOptionPane.showMessageDialog(this, "Ошибка загрузки оценок: "+ex.getMessage());
        }
    }
}