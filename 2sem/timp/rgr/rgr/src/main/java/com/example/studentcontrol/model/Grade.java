package com.example.studentcontrol.model;

import java.time.LocalDate;

public class Grade {
    private int id;
    private Student student;
    private Subject subject;
    private int grade;
    private LocalDate dateAssigned;

    public Grade() {}
    public Grade(int id, Student student, Subject subject, int grade, LocalDate dateAssigned) {
        this.id = id; this.student = student; this.subject = subject; this.grade = grade; this.dateAssigned = dateAssigned;
    }
    public int getId() { return id; }
    public void setId(int id) { this.id = id; }
    public Student getStudent() { return student; }
    public void setStudent(Student student) { this.student = student; }
    public Subject getSubject() { return subject; }
    public void setSubject(Subject subject) { this.subject = subject; }
    public int getGrade() { return grade; }
    public void setGrade(int grade) { this.grade = grade; }
    public LocalDate getDateAssigned() { return dateAssigned; }
    public void setDateAssigned(LocalDate dateAssigned) { this.dateAssigned = dateAssigned; }
}