package com.salah.gpt;

import android.content.Context;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SalahModel {

    private static final int SEQ_LEN = 256;
    private static final int VOCAB = 274;
    private static final int MAX_NEW_TOKENS = 60;
    private static final int NUM_THREADS = 4;

    // إعدادات الاختيار: تقدر تبدلهم باش تجرب
    private static final float TEMPERATURE = 0.5f;
    private static final int TOP_K = 5;
    private static final float REPETITION_PENALTY = 1.2f;
    private static final int RECENT_WINDOW = 30;

    private final Interpreter interpreter;
    private final SalahTokenizer tokenizer;
    private final Random random = new Random();

    public SalahModel(Context context) throws Exception {
        tokenizer = new SalahTokenizer(context);
        File modelFile = copyAssetToCache(context, "salah_gpt_v2.tflite");

        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(NUM_THREADS);
        interpreter = new Interpreter(modelFile, options);
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
            int recentFrom = Math.max(0, generated.size() - RECENT_WINDOW);
            List<Integer> recent = generated.subList(recentFrom, generated.size());

            int next = sample(output[0][pos], recent);

            if (next == SalahTokenizer.EOS) break;

            ids.add(next);
            generated.add(next);
        }

        return tokenizer.decode(generated);
    }

    // اختيار عشوائي من أحسن TOP_K رمز، مع temperature وrepetition penalty
    private int sample(float[] logits, List<Integer> recent) {
        float[] l = logits.clone();

        // تقليل احتمال الرموز اللي ظهرت مؤخراً
        for (int tok : recent) {
            if (tok < 0 || tok >= VOCAB) continue;
            if (l[tok] > 0) {
                l[tok] /= REPETITION_PENALTY;
            } else {
                l[tok] *= REPETITION_PENALTY;
            }
        }

        // منع الرموز الخاصة من الظهور في الجواب
        l[SalahTokenizer.PAD] = -Float.MAX_VALUE;
        l[SalahTokenizer.UNK] = -Float.MAX_VALUE;
        l[SalahTokenizer.USER] = -Float.MAX_VALUE;
        l[SalahTokenizer.ASSISTANT] = -Float.MAX_VALUE;

        // نجيبو أحسن TOP_K رمز
        boolean[] taken = new boolean[VOCAB];
        int[] topIdx = new int[TOP_K];
        int n = 0;
        for (int k = 0; k < TOP_K; k++) {
            int best = -1;
            float bestVal = -Float.MAX_VALUE;
            for (int i = 0; i < VOCAB; i++) {
                if (!taken[i] && l[i] > bestVal) {
                    bestVal = l[i];
                    best = i;
                }
            }
            if (best < 0) break;
            taken[best] = true;
            topIdx[n++] = best;
        }

        // softmax مع temperature على الرموز المختارة
        double max = l[topIdx[0]];
        double[] p = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            p[i] = Math.exp((l[topIdx[i]] - max) / TEMPERATURE);
            sum += p[i];
        }

        // اختيار عشوائي حسب الاحتمالات
        double r = random.nextDouble() * sum;
        double acc = 0;
        for (int i = 0; i < n; i++) {
            acc += p[i];
            if (acc >= r) return topIdx[i];
        }
        return topIdx[n - 1];
    }

    public void close() {
        interpreter.close();
    }
                }
