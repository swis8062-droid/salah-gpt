package com.salah.gpt;

import android.content.Context;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class SalahModel {

    private static final int SEQ_LEN = 256;
    private static final int VOCAB = 279;
    private static final int MAX_NEW_TOKENS = 200;

    private final Interpreter interpreter;
    private final SalahTokenizer tokenizer;

    public SalahModel(Context context) throws Exception {
        tokenizer = new SalahTokenizer(context);
        File modelFile = copyAssetToCache(context, "salah_gpt_v2.tflite");
        interpreter = new Interpreter(modelFile);
    }

    // نسخ الموديل من assets لمجلد الكاش، باش Interpreter يقرا ملف حقيقي
    private File copyAssetToCache(Context context, String name) throws Exception {
        File out = new File(context.getCacheDir(), name);
        if (out.exists() && out.length() > 0) return out;

        InputStream in = context.getAssets().open(name);
        OutputStream os = new FileOutputStream(out);
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) > 0) {
            os.write(buf, 0, n);
        }
        os.close();
        in.close();
        return out;
    }

    public String generate(String question) {
        List<Integer> ids = new ArrayList<>();
        ids.add(SalahTokenizer.USER);
        ids.addAll(tokenizer.encode(question.trim()));
        ids.add(SalahTokenizer.ASSISTANT);

        if (ids.size() >= SEQ_LEN) {
            return "السؤال طويل بزاف، قصّرو شوية.";
        }

        List<Integer> generated = new ArrayList<>();
        int[][] input = new int[1][SEQ_LEN];
        float[][][] output = new float[1][SEQ_LEN][VOCAB];

        for (int step = 0; step < MAX_NEW_TOKENS && ids.size() < SEQ_LEN; step++) {
            // نعبّي المدخل بـ PAD (0) ونحط فوقو الرموز الحالية
            for (int i = 0; i < SEQ_LEN; i++) {
                input[0][i] = 0;
            }
            for (int i = 0; i < ids.size(); i++) {
                input[0][i] = ids.get(i);
            }

            interpreter.run(input, output);

            // النموذج كيتوقع الرمز الجاي من الموضع الأخير الحقيقي
            int pos = ids.size() - 1;
            int next = argmax(output[0][pos]);

            if (next == SalahTokenizer.EOS) break;

            ids.add(next);
            generated.add(next);
        }

        return tokenizer.decode(generated);
    }

    // نختار أعلى قيمة، ونتجاهل الرموز الخاصة اللي ما خاصهاش تظهر في الجواب
    private int argmax(float[] logits) {
        int best = -1;
        float bestVal = -Float.MAX_VALUE;
        for (int i = 0; i < logits.length; i++) {
            if (i == SalahTokenizer.PAD
                    || i == SalahTokenizer.UNK
                    || i == SalahTokenizer.USER
                    || i == SalahTokenizer.ASSISTANT) {
                continue;
            }
            if (logits[i] > bestVal) {
                bestVal = logits[i];
                best = i;
            }
        }
        return best;
    }

    public void close() {
        interpreter.close();
    }
                                      }
