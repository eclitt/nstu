package com.example.studentcontrol.dao;

import com.example.studentcontrol.model.Subject;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public class SubjectDAO {
    public List<Subject> findAll() throws SQLException {
        List<Subject> list = new ArrayList<>();
        String sql = "SELECT * FROM subjects";
        try (Connection conn = DBConnection.getConnection();
             Statement st = conn.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            while (rs.next()) {
                Subject s = new Subject(
                    rs.getInt("id"),
                    rs.getString("name"),
                    rs.getInt("semester")
                );
                list.add(s);
            }
        }
        return list;
    }
    public void insert(Subject s) throws SQLException {
        String sql = "INSERT INTO subjects(name, semester) VALUES(?,?)";
        try (Connection conn = DBConnection.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, s.getName());
            ps.setInt(2, s.getSemester());
            ps.executeUpdate();
        }
    }
    public void update(Subject s) throws SQLException {
        String sql = "UPDATE subjects SET name=?, semester=? WHERE id=?";
        try (Connection conn = DBConnection.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, s.getName());
            ps.setInt(2, s.getSemester());
            ps.setInt(3, s.getId());
            ps.executeUpdate();
        }
    }
    public void delete(int id) throws SQLException {
        String sql = "DELETE FROM subjects WHERE id=?";
        try (Connection conn = DBConnection.getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setInt(1, id);
            ps.executeUpdate();
        }
    }
}