module com.example.ru.nstu._practice {
    requires javafx.controls;
    requires javafx.fxml;


    opens com.example.ru.nstu._practice to javafx.fxml;
    exports com.example.ru.nstu._practice;
}