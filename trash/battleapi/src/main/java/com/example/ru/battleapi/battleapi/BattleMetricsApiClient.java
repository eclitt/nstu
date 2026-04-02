package com.example.ru.battleapi.battleapi;
import org.json.JSONArray;
import org.json.JSONObject;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class BattleMetricsApiClient {

    private static final String API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbiI6Ijc4MGUxY2QyNTY0ZGQ2NjQiLCJpYXQiOjE3NzM5OTUzMTMsIm5iZiI6MTc3Mzk5NTMxMywiaXNzIjoiaHR0cHM6Ly93d3cuYmF0dGxlbWV0cmljcy5jb20iLCJzdWIiOiJ1cm46dXNlcjoxMTcxNTYwIn0.gH7ZTPW-hWK6zsUNWim6hFOCk2hoeIYfRU1HGZogCjI"; // Замените на ваш ключ
    private static final String BASE_URL = "https://api.battlemetrics.com/players";

    private final HttpClient httpClient;

    public BattleMetricsApiClient() {
        if (API_KEY == null || API_KEY.isEmpty() || API_KEY.equals("ВАШ_КЛЮЧ_ЗДЕСЬ")) {
            throw new IllegalStateException("Пожалуйста, укажите корректный API ключ в коде!");
        }
        this.httpClient = HttpClient.newHttpClient();
    }

    /**
     * Поиск игрока по имени (без include параметров)
     */
    public JSONObject searchPlayer(String playerName) throws Exception {
        String url = BASE_URL + "?filter[search]=" + playerName.replace(" ", "%20");

        System.out.println("Запрос к API: " + url);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Authorization", "Bearer " + API_KEY)
                .header("Accept", "application/json")
                .GET()
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        System.out.println("Статус ответа: " + response.statusCode());

        if (response.statusCode() == 200) {
            return new JSONObject(response.body());
        } else {
            throw new RuntimeException("Ошибка API: " + response.statusCode() + " - " + response.body());
        }
    }

    /**
     * Поиск игрока с правильными include параметрами
     * Доступные include: server, identifier, playerFlag, flagPlayer
     */
    public JSONObject searchPlayerWithIncludes(String playerName, String includes) throws Exception {
        String url = BASE_URL + "?filter[search]=" + playerName.replace(" ", "%20");

        if (includes != null && !includes.isEmpty()) {
            url += "&include=" + includes;
        }

        System.out.println("Запрос к API: " + url);

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Authorization", "Bearer " + API_KEY)
                .header("Accept", "application/json")
                .GET()
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 200) {
            return new JSONObject(response.body());
        } else {
            throw new RuntimeException("Ошибка API: " + response.statusCode() + " - " + response.body());
        }
    }

    /**
     * Получение информации об игроке по ID
     */
    public JSONObject getPlayerById(String playerId) throws Exception {
        String url = BASE_URL + "/" + playerId;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Authorization", "Bearer " + API_KEY)
                .header("Accept", "application/json")
                .GET()
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 200) {
            return new JSONObject(response.body());
        } else {
            throw new RuntimeException("Ошибка API: " + response.statusCode());
        }
    }

    /**
     * Получение информации об игроке с дополнительными данными
     * @param playerId ID игрока
     * @param includes список include через запятую (server, identifier, playerFlag, flagPlayer)
     */
    public JSONObject getPlayerByIdWithIncludes(String playerId, String includes) throws Exception {
        String url = BASE_URL + "/" + playerId;

        if (includes != null && !includes.isEmpty()) {
            url += "?include=" + includes;
        }

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Authorization", "Bearer " + API_KEY)
                .header("Accept", "application/json")
                .GET()
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 200) {
            return new JSONObject(response.body());
        } else {
            throw new RuntimeException("Ошибка API: " + response.statusCode());
        }
    }

    /**
     * Поиск игрока с лимитом результатов
     */
    public JSONObject searchPlayerWithLimit(String playerName, int limit) throws Exception {
        String url = BASE_URL + "?filter[search]=" + playerName.replace(" ", "%20")
                + "&page[limit]=" + limit;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Authorization", "Bearer " + API_KEY)
                .header("Accept", "application/json")
                .GET()
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() == 200) {
            return new JSONObject(response.body());
        } else {
            throw new RuntimeException("Ошибка API: " + response.statusCode());
        }
    }
}