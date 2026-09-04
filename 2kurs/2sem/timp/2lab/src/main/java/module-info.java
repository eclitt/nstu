module com.example.ru.nstu._1lab {
    requires javafx.controls;
    requires javafx.fxml;


    opens com.example.ru.nstu._1lab to javafx.fxml;
    exports com.example.ru.nstu._1lab;
}