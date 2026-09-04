CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    group_name VARCHAR(20),
    enrollment_date DATE
);

CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    semester INT
);

CREATE TABLE IF NOT EXISTS grades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INT REFERENCES students(id) ON DELETE CASCADE,
    subject_id INT REFERENCES subjects(id) ON DELETE CASCADE,
    grade INT CHECK (grade BETWEEN 1 AND 10),
    date_assigned DATE DEFAULT CURRENT_DATE
);

CREATE INDEX IF NOT EXISTS idx_grades_student_id ON grades(student_id);
CREATE INDEX IF NOT EXISTS idx_grades_subject_id ON grades(subject_id);