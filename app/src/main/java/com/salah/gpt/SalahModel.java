package com.salah.gpt;

import android.content.Context;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class SalahModel {

    // بدّل هنا فقط إلا جيتي بنسخة جديدة من الموديل
    private static final String MODEL_ASSET = "salah_gpt_v3.tflite";

    private static final int MAX_NEW_TOKENS = 60;
    private static final int NUM_THREADS = 2;

    private static final float TEMPERATURE = 0.5f;
    private static final int TOP_K = 5;
    private static final float REPETITION_PENALTY = 1.2f;
    private static final int RECENT_WINDOW = 30;

    private static final String MSG_EMPTY_QUESTION = "كتب ليا سؤال الأول.";
    private static final String MSG_TOO_LONG = "السؤال طويل بزاف، قصّرو شوية.";
    private static final String MSG_NO_ANSWER = "ما قدرتش نولّد جواب. جرّب سؤال آخر.";

    private final Interpreter interpreter;
    private final SalahTokenizer tokenizer;
    private final Random random = new Random();

    private final int seqLen;
    private final int vocabSize;

    public SalahModel(Context context) throws Exception {
        tokenizer = new SalahTokenizer(context);

        File modelFile = copyAssetToCache(context, MODEL_ASSET);

        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(NUM_THREADS);

        Interpreter tempInterpreter = new Interpreter(modelFile, options);
        int[] dims;

        try {
            dims = validateModel(tempInterpreter);
        } catch (Exception e) {
            tempInterpreter.close();
            throw e;
        }

        interpreter = tempInterpreter;
        seqLen = dims[0];
        vocabSize = dims[1];
    }

    // يتحقق من شكل الموديل ونوعو، ويرجع [seqLen, vocabSize]
    private static int[] validateModel(Interpreter interp) {
        interp.allocateTensors();

        Tensor input = interp.getInputTensor(0);
        Tensor output = interp.getOutputTensor(0);

        int[] inShape = input.shape();
        int[] outShape = output.shape();

        if (input.dataType() != DataType.INT32) {
            throw new IllegalStateException("Input must be INT32, got " + input.dataType());
        }
        if (output.dataType() != DataType.FLOAT32) {
            throw new IllegalStateException("Output must be FLOAT32, got " + output.dataType());
        }
        if (inShape.length != 2 || inShape[0] != 1) {
            throw new IllegalStateException("Invalid input shape: " + Arrays.toString(inShape));
        }
        if (outShape.length != 3 || outShape[0] != 1 || outShape[1] != inShape[1]) {
            throw new IllegalStateException("Invalid output shape: " + Arrays.toString(outShape));
        }

        int seq = inShape[1];
        int vocab = outShape[2];

        if (seq <= 0 || vocab <= SalahTokenizer.EOS) {
            throw new IllegalStateException("Invalid model dimensions");
        }

        return new int[]{seq, vocab};
    }

    private File copyAssetToCache(Context context, String name) throws Exception {
        File out = new File(context.getCacheDir(), name);
        File temp = new File(context.getCacheDir(), name + ".tmp");

        if (temp.exists() && !temp.delete()) {
            throw new IllegalStateException("Cannot clear temp model file");
        }

        try (InputStream in = context.getAssets().open(name);
             FileOutputStream os = new FileOutputStream(temp)) {

            byte[] buffer = new byte[8192];
            int count;
            while ((count = in.read(buffer)) != -1) {
                os.write(buffer, 0, count);
            }
            os.flush();
            os.getFD().sync();
        }

        if (out.exists() && !out.delete()) {
            temp.delete();
            throw new IllegalStateException("Cannot replace cached model");
        }

        if (!temp.renameTo(out)) {
            temp.delete();
            throw new IllegalStateException("Cannot move model to cache");
        }

        if (!out.exists() || out.length() == 0) {
            throw new IllegalStateException("Model file is empty");
        }

        return out;
    }

    public String generate(String question) {
        if (question == null || question.trim().isEmpty()) {
            return MSG_EMPTY_QUESTION;
        }

        List<Integer> ids = new ArrayList<>();
        ids.add(SalahTokenizer.USER);
        ids.addAll(tokenizer.encode(question.trim()));
        ids.add(SalahTokenizer.ASSISTANT);

        if (ids.size() >= seqLen) {
            return MSG_TOO_LONG;
        }

        List<Integer> generated = new ArrayList<>();
        int[][] input = new int[1][seqLen];
        float[][][] output = new float[1][seqLen][vocabSize];

        for (int step = 0; step < MAX_NEW_TOKENS && ids.size() < seqLen; step++) {

            Arrays.fill(input[0], SalahTokenizer.PAD);
            for (int i = 0; i < ids.size(); i++) {
                input[0][i] = ids.get(i);
            }

            interpreter.run(input, output);

            // الـ logits اللي كنحتاجوهم هو آخر توكن فالسياق الحالي
            int position = ids.size() - 1;

            int from = Math.max(0, generated.size() - RECENT_WINDOW);
            List<Integer> recent = generated.subList(from, generated.size());

            int next = sample(output[0][position], recent);

            if (next == SalahTokenizer.EOS) {
                break;
            }

            ids.add(next);
            generated.add(next);
        }

        String answer = tokenizer.decode(generated).trim();
        return answer.isEmpty() ? MSG_NO_ANSWER : answer;
    }

    private int sample(float[] logits, List<Integer> recent) {
        if (logits == null || logits.length != vocabSize) {
            return SalahTokenizer.EOS;
        }

        float[] values = logits.clone();

        // عقوبة التكرار: كل توكن ظهر مؤخراً كيتنقص احتمالو مرة وحدة
        boolean[] penalized = new boolean[vocabSize];
        for (int token : recent) {
            if (token < 0 || token >= vocabSize || penalized[token]) {
                continue;
            }
            penalized[token] = true;

            if (values[token] > 0) {
                values[token] /= REPETITION_PENALTY;
            } else {
                values[token] *= REPETITION_PENALTY;
            }
        }

        // التوكنات الخاصة ما يمكن تتولدش
        values[SalahTokenizer.PAD] = Float.NEGATIVE_INFINITY;
        values[SalahTokenizer.UNK] = Float.NEGATIVE_INFINITY;
        values[SalahTokenizer.USER] = Float.NEGATIVE_INFINITY;
        values[SalahTokenizer.ASSISTANT] = Float.NEGATIVE_INFINITY;

        int k = Math.min(TOP_K, vocabSize);
        int[] top = new int[k];
        boolean[] selected = new boolean[vocabSize];
        int count = 0;

        // نختارو top-k بدون sort كامل
        for (int n = 0; n < k; n++) {
            int best = -1;
            float bestValue = Float.NEGATIVE_INFINITY;

            for (int i = 0; i < vocabSize; i++) {
                float value = values[i];
                if (!selected[i] && !Float.isNaN(value) && value > bestValue) {
                    bestValue = value;
                    best = i;
                }
            }

            if (best < 0) break;

            selected[best] = true;
            top[count++] = best;
        }

        if (count == 0) {
            return SalahTokenizer.EOS;
        }

        // softmax مع temperature، وبطريقة مستقرة عددياً
        double max = values[top[0]];
        double[] probabilities = new double[count];
        double sum = 0;

        for (int i = 0; i < count; i++) {
            double p = Math.exp((values[top[i]] - max) / TEMPERATURE);
            if (Double.isNaN(p) || Double.isInfinite(p)) {
                p = 0;
            }
            probabilities[i] = p;
            sum += p;
        }

        if (sum <= 0 || Double.isInfinite(sum)) {
            return top[0];
        }

        // اختيار عشوائي حسب الاحتمالات
        double r = random.nextDouble() * sum;
        double accumulated = 0;

        for (int i = 0; i < count; i++) {
            accumulated += probabilities[i];
            if (accumulated >= r) {
                return top[i];
            }
        }

        return top[count - 1];
    }

    public void close() {
        interpreter.close();
    }
            }
