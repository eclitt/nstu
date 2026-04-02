module com.eclitt.chat.rgz_chat {
    requires javafx.controls;
    requires javafx.fxml;
    requires javafx.web;

    requires org.controlsfx.controls;
    requires net.synedra.validatorfx;
    requires org.kordamp.ikonli.javafx;
    requires org.kordamp.bootstrapfx.core;
   // requires eu.hansolo.tilesfx;
    requires com.almasb.fxgl.all;

    opens com.eclitt.chat.rgz_chat to javafx.fxml;
    exports com.eclitt.chat.rgz_chat;
}