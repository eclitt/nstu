package com.example.studentcontrol.dao;

import com.example.studentcontrol.model.Grade;
import com.example.studentcontrol.model.Student;
import com.example.studentcontrol.model.Subject;
import java.sql.*;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

public class GradeDAO {
    public List<Grade> findByStudent(int studentId) throws SQLException {
        List<Grade> list = new ArrayList<>();
        String sql = "SELECT g.id, g.grade, g.date_assigned, " +
                     "s.id AS sid, s.first_name, s.last_name, s.group_name, s.enrollment_date, " +
                     "sub.id AS subid, sub.name, sub.semester " +
                     "FROM grades g " +
                     "JOIN students s ON g.student_id=s.id " +
                     "JOIN subjects sub ON g.subject_id=sub.id " +
                     "WHERE s.id=?";
        try (Connection conn = DBConnection.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setInt(1, studentId);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    Student st = new Student(
                        rs.getInt("sid"),
                        rs.getString("first_name"),
                        rs.getString("last_name"),
                        rs.getString("group_name"),
                        rs.getDate("enrollment_date").toLocalDate()
                    );
                    Subject sub = new Subject(
                        rs.getInt("subid"),
                        rs.getString("name"),
                        rs.getInt("semester")
                    );
                    Grade gr = new Grade(
                        rs.getInt("id"),
                        st,
                        sub,
                        rs.getInt("grade"),
                        rs.getDate("date_assigned").toLocalDate()
                    );
                    list.add(gr);
                }
            }
        }
        return list;
    }
    public void insert(int studentId, int subjectId, int grade) throws SQLException {
        String sql = "INSERT INTO grades(student_id, subject_id, grade, date_assigned) VALUES(?,?,?,?)";
        try (Connection conn = DBConnection.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setInt(1, studentId);
            ps.setInt(2, subjectId);
            ps.setInt(3, grade);
            ps.setDate(4, Date.valueOf(LocalDate.now()));
            ps.executeUpdate();
        }
    }
    public void delete(int id) throws SQLException {
        String sql = "DELETE FROM grades WHERE id=?";
        try (Connection conn = DBConnection.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setInt(1, id);
            ps.executeUpdate();
        }
    }
}