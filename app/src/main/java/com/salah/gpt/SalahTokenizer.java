package com.salah.gpt;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SalahTokenizer {

    // نفس الـ ids ديال Python (SPECIALS ترتيبهم هو هاد)
    public static final int PAD = 0;
    public static final int UNK = 1;
    public static final int USER = 2;
    public static final int ASSISTANT = 3;
    public static final int EOS = 4;

    private static final int NUM_SPECIALS = 5;
    private static final String ASSET = "salah_tokenizer.json";

    // نفس الـ regex ديال ByteLevel (GPT-2)
    private static final Pattern PRE_TOKENIZE = Pattern.compile(
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    );

    // جدول byte -> char ديال GPT-2
    private static final char[] BYTE_TO_CHAR = new char[256];
    private static final int[] CHAR_TO_BYTE = new int[512];

    static {
        List<Integer> bs = new ArrayList<>();
        for (int b = '!'; b <= '~'; b++) bs.add(b);
        for (int b = 0xA1; b <= 0xAC; b++) bs.add(b);
        for (int b = 0xAE; b <= 0xFF; b++) bs.add(b);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }
        for (int i = 0; i < bs.size(); i++) {
            BYTE_TO_CHAR[bs.get(i)] = (char) cs.get(i).intValue();
            CHAR_TO_BYTE[cs.get(i)] = bs.get(i);
        }
    }

    private final Map<String, Integer> vocab = new HashMap<>();
    private final Map<Integer, String> idToToken = new HashMap<>();
    private final Map<String, Integer> merges = new HashMap<>();
    private final Map<String, List<Integer>> cache = new HashMap<>();
    private final boolean addPrefixSpace;

    public SalahTokenizer(Context context) throws Exception {
        JSONObject root = new JSONObject(readAsset(context, ASSET));
        JSONObject model = root.getJSONObject("model");

        JSONObject vocabJson = model.getJSONObject("vocab");
        Iterator<String> keys = vocabJson.keys();
        while (keys.hasNext()) {
            String token = keys.next();
            int id = vocabJson.getInt(token);
            vocab.put(token, id);
            idToToken.put(id, token);
        }

        // merges ممكن تكون "a b" أو ["a", "b"] حسب نسخة tokenizers
        JSONArray mergesJson = model.getJSONArray("merges");
        for (int i = 0; i < mergesJson.length(); i++) {
            Object item = mergesJson.get(i);
            String a;
            String b;
            if (item instanceof JSONArray) {
                a = ((JSONArray) item).getString(0);
                b = ((JSONArray) item).getString(1);
            } else {
                String s = (String) item;
                int sp = s.indexOf(' ');
                a = s.substring(0, sp);
                b = s.substring(sp + 1);
            }
            merges.put(a + " " + b, i);
        }

        JSONObject pre = root.optJSONObject("pre_tokenizer");
        addPrefixSpace = pre != null && pre.optBoolean("add_prefix_space", false);
    }

    public List<Integer> encode(String text) {
        String cleaned = clean(text);
        if (cleaned.isEmpty()) {
            return new ArrayList<>();
        }
        if (addPrefixSpace) {
            cleaned = " " + cleaned;
        }

        List<Integer> ids = new ArrayList<>();
        Matcher m = PRE_TOKENIZE.matcher(cleaned);
        while (m.find()) {
            String piece = m.group();
            StringBuilder mapped = new StringBuilder();
            for (byte b : piece.getBytes(StandardCharsets.UTF_8)) {
                mapped.append(BYTE_TO_CHAR[b & 0xFF]);
            }
            ids.addAll(bpe(mapped.toString()));
        }
        return ids;
    }

    public String decode(List<Integer> ids) {
        StringBuilder mapped = new StringBuilder();
        for (int id : ids) {
            if (id < NUM_SPECIALS) continue;
            String token = idToToken.get(id);
            if (token != null) {
                mapped.append(token);
            }
        }

        byte[] bytes = new byte[mapped.length()];
        for (int i = 0; i < mapped.length(); i++) {
            bytes[i] = (byte) CHAR_TO_BYTE[mapped.charAt(i)];
        }
        return new String(bytes, StandardCharsets.UTF_8).trim();
    }

    // نفس تنظيف النص اللي استعملناه فـ Python قبل التدريب
    private static String clean(String s) {
        String out = s.trim().replaceAll("\\s+", " ");
        out = out.replace(" ’ ", "’").replace(" ' ", "'");
        out = out.replaceAll("\\s+([,.!?;:])", "$1");
        return out.trim();
    }

    private List<Integer> bpe(String token) {
        List<Integer> cached = cache.get(token);
        if (cached != null) {
            return cached;
        }

        List<String> parts = new ArrayList<>();
        for (int i = 0; i < token.length(); i++) {
            parts.add(String.valueOf(token.charAt(i)));
        }

        while (parts.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            int bestIdx = -1;

            for (int i = 0; i < parts.size() - 1; i++) {
                Integer rank = merges.get(parts.get(i) + " " + parts.get(i + 1));
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }

            if (bestIdx < 0) break;

            parts.set(bestIdx, parts.get(bestIdx) + parts.get(bestIdx + 1));
            parts.remove(bestIdx + 1);
        }

        List<Integer> ids = new ArrayList<>(parts.size());
        for (String p : parts) {
            Integer id = vocab.get(p);
            ids.add(id != null ? id : UNK);
        }

        cache.put(token, ids);
        return ids;
    }

    private static String readAsset(Context context, String name) throws Exception {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(context.getAssets().open(name), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append('\n');
            }
        }
        return sb.toString();
    }
             }
