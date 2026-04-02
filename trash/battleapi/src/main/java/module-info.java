module com.example.ru.battleapi.battleapi {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.kordamp.bootstrapfx.core;
    requires org.json;
    requires java.net.http;
    opens com.example.ru.battleapi.battleapi to javafx.fxml;
    exports com.example.ru.battleapi.battleapi;
}