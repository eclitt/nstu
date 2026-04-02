package com.example.ru.battleapi.battleapi;
import org.json.JSONArray;
import org.json.JSONObject;
import java.util.Scanner;

public class BattleMetricsApp {

    private final BattleMetricsApiClient apiClient;

    public BattleMetricsApp() {
        this.apiClient = new BattleMetricsApiClient();
    }

    /**
     * Выводит информацию об игроке в читаемом формате
     */
    public void displayPlayerInfo(JSONObject searchResult) {
        JSONArray data = searchResult.getJSONArray("data");

        if (data.length() == 0) {
            System.out.println("Игрок не найден!");
            return;
        }

        System.out.println("Найдено игроков: " + data.length());
        System.out.println("=================================");

        for (int i = 0; i < data.length(); i++) {
            JSONObject player = data.getJSONObject(i);
            JSONObject attributes = player.getJSONObject("attributes");

            System.out.println("\nИгрок #" + (i + 1));
            System.out.println("ID: " + player.getString("id"));
            System.out.println("Имя: " + attributes.getString("name"));
            System.out.println("Ссылка: https://www.battlemetrics.com/players/" + player.getString("id"));

            // Проверяем наличие дополнительных полей
            if (attributes.has("createdAt")) {
                System.out.println("Создан: " + attributes.getString("createdAt"));
            }

            if (attributes.has("updatedAt")) {
                System.out.println("Обновлен: " + attributes.getString("updatedAt"));
            }

            System.out.println("---------------------------------");
        }
    }

    /**
     * Выводит краткую информацию об игроке
     */
    public void displayPlayerInfoShort(JSONObject searchResult) {
        JSONArray data = searchResult.getJSONArray("data");

        if (data.length() == 0) {
            System.out.println("Игрок не найден!");
            return;
        }

        System.out.println("\n=== Результаты поиска ===");
        System.out.println("Найдено игроков: " + data.length());

        for (int i = 0; i < data.length(); i++) {
            JSONObject player = data.getJSONObject(i);
            JSONObject attributes = player.getJSONObject("attributes");

            System.out.println((i + 1) + ". " + attributes.getString("name") + " (ID: " + player.getString("id") + ")");
        }
    }

    /**
     * Показывает детальную информацию об игроке
     */
    public void showPlayerDetails(String playerId) {
        try {
            JSONObject playerDetails = apiClient.getPlayerById(playerId);
            System.out.println("\n=== Детальная информация ===");

            JSONObject data = playerDetails.getJSONObject("data");
            JSONObject attributes = data.getJSONObject("attributes");

            System.out.println("ID: " + data.getString("id"));
            System.out.println("Имя: " + attributes.getString("name"));
            System.out.println("Ссылка: https://www.battlemetrics.com/players/" + data.getString("id"));

            if (attributes.has("createdAt")) {
                System.out.println("Создан: " + attributes.getString("createdAt"));
            }

            if (attributes.has("updatedAt")) {
                System.out.println("Обновлен: " + attributes.getString("updatedAt"));
            }

            // Выводим полный JSON для отладки
            System.out.println("\nПолный ответ API:");
            System.out.println(playerDetails.toString(2));

        } catch (Exception e) {
            System.err.println("Ошибка при получении детальной информации: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Запуск интерактивного режима
     */
    public void runInteractiveMode() {
        Scanner scanner = new Scanner(System.in);

        System.out.println("=== BattleMetrics Поиск игрока ===");
        System.out.println("1. Поиск по имени");
        System.out.println("2. Поиск по имени (краткий вывод)");
        System.out.println("3. Поиск с лимитом результатов");
        System.out.println("4. Выход");
        System.out.print("Выберите опцию: ");

        String choice = scanner.nextLine();

        switch (choice) {
            case "1":
                System.out.print("Введите имя игрока: ");
                String name = scanner.nextLine();
                try {
                    JSONObject result = apiClient.searchPlayer(name);
                    displayPlayerInfo(result);
                } catch (Exception e) {
                    System.err.println("Ошибка: " + e.getMessage());
                }
                break;

            case "2":
                System.out.print("Введите имя игрока: ");
                String shortName = scanner.nextLine();
                try {
                    JSONObject result = apiClient.searchPlayer(shortName);
                    displayPlayerInfoShort(result);

                    // Предлагаем посмотреть детальную информацию
                    JSONArray players = result.getJSONArray("data");
                    if (players.length() > 0) {
                        System.out.print("\nВведите ID игрока для детальной информации (или Enter для пропуска): ");
                        String playerId = scanner.nextLine();
                        if (!playerId.isEmpty()) {
                            showPlayerDetails(playerId);
                        }
                    }

                } catch (Exception e) {
                    System.err.println("Ошибка: " + e.getMessage());
                }
                break;

            case "3":
                System.out.print("Введите имя игрока: ");
                String limitName = scanner.nextLine();
                System.out.print("Введите лимит результатов (1-100): ");
                int limit = Integer.parseInt(scanner.nextLine());
                try {
                    JSONObject result = apiClient.searchPlayerWithLimit(limitName, limit);
                    displayPlayerInfoShort(result);
                } catch (Exception e) {
                    System.err.println("Ошибка: " + e.getMessage());
                }
                break;

            case "4":
                System.out.println("До свидания!");
                break;

            default:
                System.out.println("Неверная опция!");
        }

        scanner.close();
    }

    /**
     * Простой режим поиска с аргументами командной строки
     */
    public void runSimpleMode(String playerName) {
        try {
            JSONObject result = apiClient.searchPlayer(playerName);
            displayPlayerInfoShort(result);

            // Если нашли игроков, предлагаем посмотреть детальную информацию
            JSONArray players = result.getJSONArray("data");
            if (players.length() > 0) {
                Scanner scanner = new Scanner(System.in);
                System.out.print("\nПоказать детальную информацию? (y/n): ");
                String answer = scanner.nextLine();

                if (answer.equalsIgnoreCase("y")) {
                    String playerId = players.getJSONObject(0).getString("id");
                    showPlayerDetails(playerId);
                }
                scanner.close();
            }

        } catch (Exception e) {
            System.err.println("Ошибка при поиске: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        BattleMetricsApp app = new BattleMetricsApp();

        if (args.length > 0) {
            // Режим с аргументами командной строки
            app.runSimpleMode(args[0]);
        } else {
            // Интерактивный режим
            app.runInteractiveMode();
        }
    }
}