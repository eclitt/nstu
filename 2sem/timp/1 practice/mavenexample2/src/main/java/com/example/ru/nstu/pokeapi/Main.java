package com.example.ru.nstu.pokeapi;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.Charset;

public class Main {
    public static String ListUrl = "https://pokeapi.co/api/v2/pokemon/";
    public String PokemonUrl;

    public static String getJson(String _url) throws IOException {
        URL url = new URL(_url);
        URLConnection conn = url.openConnection();
        conn.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 6.1; WOW64");
        conn.connect();
        BufferedReader r = new BufferedReader(
                new InputStreamReader(
                        conn.getInputStream(),
                        Charset.forName("UTF-8")
                )
        );
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = r.readLine()) != null) {
            sb.append(line);
        }
        String json = sb.toString();
        return json;
    }

    public static void main(String[] args) {
        String json = null;

        try {
            json = getJson(ListUrl);
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            if (json == null) {
                System.out.println("empty");
            } else {
                printFirst3Pokemons(json);
                System.out.println("\n=== ALL POKEMONS ===\n");
                printAllPokemons(json);
            }
        }

    }

    public static void printFirst3Pokemons(String _json) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(_json);
            ArrayNode results = (ArrayNode) root.get("results");

            int count = Math.min(3, results.size());

            for (int i = 0; i < count; i++) {
                JsonNode pokemon = results.get(i);
                String url = pokemon.get("url").asText();
                String pokemonJson = getJson(url);
                printPokemonDetails(pokemonJson);
                System.out.println("-------------------");
            }
        } catch (JsonProcessingException ex) {
            ex.printStackTrace();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    public static void printPokemonDetails(String _json) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(_json);

            String name = root.get("name").asText();
            int height = root.get("height").asInt();
            int weight = root.get("weight").asInt();

            System.out.println("Name: " + name);
            System.out.println("Height: " + height);
            System.out.println("Weight: " + weight);

            // Types
            System.out.print("Types: ");
            ArrayNode types = (ArrayNode) root.get("types");
            for (JsonNode type : types) {
                System.out.print(type.get("type").get("name").asText() + " ");
            }
            System.out.println();

            // Stats
            System.out.println("Stats:");
            ArrayNode stats = (ArrayNode) root.get("stats");
            for (JsonNode stat : stats) {
                String statName = stat.get("stat").get("name").asText();
                int baseStat = stat.get("base_stat").asInt();
                System.out.println("  " + statName + ": " + baseStat);
            }

        } catch (JsonProcessingException ex) {
            ex.printStackTrace();
        }
    }

    public static void printAllPokemons(String _json) {
        try {
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(_json);
            ArrayNode results = (ArrayNode) root.get("results");

            for (JsonNode pokemon : results) {
                String name = pokemon.get("name").asText();
                String url = pokemon.get("url").asText();
                System.out.println(name + " - " + url);
            }
        } catch (JsonProcessingException ex) {
            ex.printStackTrace();
        }
    }

}

