package com.example.studentcontrol.dao;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.sql.*;

public class DBConnection {
    private static final String URL = "jdbc:sqlite:students.db?date_string_format=yyyy-MM-dd";

    public static Connection getConnection() throws SQLException {
        Connection conn = DriverManager.getConnection(URL);
        try (Statement stmt = conn.createStatement()) {
            stmt.execute("PRAGMA foreign_keys = ON");
        }
        return conn;
    }

    public static void initializeDatabase() throws SQLException, IOException {
        try (Connection conn = getConnection(); Statement stmt = conn.createStatement()) {
            String schema = readSchema();
            for (String command : schema.split(";")) {
                String sql = command.trim();
                if (!sql.isEmpty()) {
                    stmt.execute(sql);
                }
            }
        }
    }

    private static String readSchema() throws IOException {
        try (InputStream is = DBConnection.class.getResourceAsStream("/db/schema.sql")) {
            if (is == null) throw new IOException("schema.sql not found");
            ByteArrayOutputStream os = new ByteArrayOutputStream();
            byte[] buffer = new byte[4096];
            int len;
            while ((len = is.read(buffer)) != -1) {
                os.write(buffer, 0, len);
            }
            return os.toString(StandardCharsets.UTF_8);
        }
    }
}