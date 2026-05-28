package com.example.studentcontrol.dao;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.nio.charset.StandardCharsets;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class DBConnection {
    // date_string_format гарантирует единый формат дат для SQLite JDBC
    private static final String URL = "jdbc:sqlite:students.db?date_string_format=yyyy-MM-dd";

    public static Connection getConnection() throws SQLException {
        Connection connection = DriverManager.getConnection(URL);
        try (Statement statement = connection.createStatement()) {
            statement.execute("PRAGMA foreign_keys = ON");
        }
        return connection;
    }

    public static void initializeDatabase() throws SQLException, IOException {
        try (Connection connection = getConnection();
             Statement statement = connection.createStatement()) {
            String schema = readSchema();
            for (String command : schema.split(";")) {
                String sql = command.trim();
                if (!sql.isEmpty()) {
                    statement.execute(sql);
                }
            }
        }
    }

    private static String readSchema() throws IOException {
        try (InputStream inputStream = DBConnection.class.getResourceAsStream("/db/schema.sql")) {
            if (inputStream == null) {
                throw new IOException("Файл схемы БД не найден: /db/schema.sql");
            }
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            return new String(outputStream.toByteArray(), StandardCharsets.UTF_8);
        }
    }
}