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
import java.util.Iterator;

public class Main {
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
    public static String ListUrl = "https://pokeapi.co/api/v2/pokemon/";

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
                printPokemons(json);
            }
        }

    }
    public static void printPokemons(String _json){
        ObjectMapper mapper = new ObjectMapper();
        try {
            JsonNode root = mapper.readTree(_json);
            System.out.println(root.get("count").asInt());

            ArrayNode result = (ArrayNode) root.get("results");

            for (JsonNode current : result) {
                System.out.println(current.get("name").asText() + ": " + current.get("url").asText());
            }
        } catch (JsonProcessingException ex) {
            ex.printStackTrace();
        }
    }
}

