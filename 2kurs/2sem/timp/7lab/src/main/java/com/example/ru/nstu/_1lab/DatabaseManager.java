package com.example.ru.nstu._1lab;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Класс для работы с базой данных SQLite.
 * Обеспечивает сохранение и загрузку объектов сотрудников.
 */
public class DatabaseManager {
    protected static String DB_URL = "jdbc:sqlite:simulation.db";

    static {
        try {
            // Инициализация таблицы при первом обращении к классу
            try (Connection conn = DriverManager.getConnection(DB_URL);
                 Statement stmt = conn.createStatement()) {
                String sql = "CREATE TABLE IF NOT EXISTS employees (" +
                        "id INTEGER PRIMARY KEY," +
                        "type TEXT NOT NULL," +
                        "birthTime BIGINT," +
                        "lifetime BIGINT," +
                        "x DOUBLE," +
                        "y DOUBLE," +
                        "width DOUBLE," +
                        "height DOUBLE," +
                        "red DOUBLE," +
                        "green DOUBLE," +
                        "blue DOUBLE," +
                        "opacity DOUBLE" +
                        ")";
                stmt.execute(sql);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    /**
     * Сохраняет список сотрудников определенного типа в базу данных.
     * Перед сохранением удаляет старые записи этого типа.
     * @param employees список сотрудников
     * @param type тип сотрудников ("Developer" или "Manager")
     */
    public static void saveEmployees(List<Employee> employees, String type) {
        try (Connection conn = DriverManager.getConnection(DB_URL)) {
            conn.setAutoCommit(false);
            
            // Удаляем существующие записи этого типа
            String deleteSql = "DELETE FROM employees WHERE type = ?";
            try (PreparedStatement pstmt = conn.prepareStatement(deleteSql)) {
                pstmt.setString(1, type);
                pstmt.executeUpdate();
            }

            // Вставляем новые записи
            String insertSql = "INSERT INTO employees (id, type, birthTime, lifetime, x, y, width, height, red, green, blue, opacity) " +
                               "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
            try (PreparedStatement pstmt = conn.prepareStatement(insertSql)) {
                for (Employee emp : employees) {
                    if (emp.getType().equals(type)) {
                        pstmt.setInt(1, emp.getId());
                        pstmt.setString(2, emp.getType());
                        pstmt.setLong(3, emp.getBirthTime());
                        pstmt.setLong(4, emp.getLifetime());
                        pstmt.setDouble(5, emp.getX());
                        pstmt.setDouble(6, emp.getY());
                        pstmt.setDouble(7, emp.getWidth());
                        pstmt.setDouble(8, emp.getHeight());
                        pstmt.setDouble(9, emp.getColor().getRed());
                        pstmt.setDouble(10, emp.getColor().getGreen());
                        pstmt.setDouble(11, emp.getColor().getBlue());
                        pstmt.setDouble(12, emp.getColor().getOpacity());
                        pstmt.addBatch();
                    }
                }
                pstmt.executeBatch();
            }
            conn.commit();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    /**
     * Загружает сотрудников определенного типа из базы данных.
     * @param type тип сотрудников ("Developer" или "Manager")
     * @return список загруженных сотрудников
     */
    public static void setURL(String _DB_URL) {
            DB_URL = _DB_URL;
    }
    public static String getDbUrl() {
        return DB_URL;
    }
    public static List<Employee> loadEmployees(String type) {
        List<Employee> result = new ArrayList<>();
        String sql = "SELECT * FROM employees WHERE type = ?";
        
        try (Connection conn = DriverManager.getConnection(DB_URL);
             PreparedStatement pstmt = conn.prepareStatement(sql)) {
            pstmt.setString(1, type);
            ResultSet rs = pstmt.executeQuery();

            while (rs.next()) {
                Employee emp;
                if ("Developer".equals(type)) {
                    emp = new Developer();
                } else if ("Manager".equals(type)) {
                    emp = new Manager();
                } else {
                    continue;
                }

                emp.setId(rs.getInt("id"));
                emp.setBirthTime(rs.getLong("birthTime"));
                emp.setLifetime(rs.getLong("lifetime"));
                emp.setX(rs.getDouble("x"));
                emp.setY(rs.getDouble("y"));
                emp.setWidth(rs.getDouble("width"));
                emp.setHeight(rs.getDouble("height"));
                emp.setRGB(
                    rs.getDouble("red"),
                    rs.getDouble("green"),
                    rs.getDouble("blue"),
                    rs.getDouble("opacity")
                );
                
                result.add(emp);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return result;
    }
}
