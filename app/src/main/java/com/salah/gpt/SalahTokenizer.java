package com.salah.gpt;

import android.content.Context;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class SalahTokenizer {

    public static final int PAD = 0;
    public static final int UNK = 1;
    public static final int USER = 2;
    public static final int ASSISTANT = 3;
    public static final int EOS = 4;

    private final Map<String, Integer> charToId = new HashMap<>();
    private final Map<Integer, String> idToChar = new HashMap<>();

    public SalahTokenizer(Context context) throws Exception {
        InputStream is = context.getAssets().open("salah_tokenizer.json");
        BufferedReader br = new BufferedReader(new InputStreamReader(is, "UTF-8"));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = br.readLine()) != null) {
            sb.append(line);
        }
        br.close();

        // القاموس ملف JSON فيه: "الحرف": رقم
        JSONObject obj = new JSONObject(sb.toString());
        Iterator<String> keys = obj.keys();
        while (keys.hasNext()) {
            String key = keys.next();
            int id = obj.getInt(key);
            charToId.put(key, id);
            idToChar.put(id, key);
        }
    }

    // تحويل النص إلى قائمة أرقام (بدون padding)
    public List<Integer> encode(String text) {
        List<Integer> ids = new ArrayList<>();
        for (int i = 0; i < text.length(); i++) {
            String ch = String.valueOf(text.charAt(i));
            Integer id = charToId.get(ch);
            ids.add(id != null ? id : UNK);
        }
        return ids;
    }

    // تحويل قائمة أرقام إلى نص (كنتجاهلو الرموز الخاصة)
    public String decode(List<Integer> ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            if (id <= EOS) continue; // PAD, UNK, USER, ASSISTANT, EOS
            String ch = idToChar.get(id);
            if (ch != null) sb.append(ch);
        }
        return sb.toString();
    }
      }
