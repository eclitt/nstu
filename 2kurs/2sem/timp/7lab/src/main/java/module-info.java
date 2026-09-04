module com.example.ru.nstu._1lab {
    requires javafx.controls;
    requires javafx.fxml;
    requires java.sql;


    opens com.example.ru.nstu._1lab to javafx.fxml;
    exports com.example.ru.nstu._1lab;
}