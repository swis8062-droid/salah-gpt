package com.salah.gpt;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.TextView;

public class MainActivity extends Activity {

    private TextView chatText;
    private EditText inputText;
    private ScrollView scrollView;
    private Button sendButton;

    private volatile SalahModel model;
    private final Handler handler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        chatText = findViewById(R.id.chatText);
        inputText = findViewById(R.id.inputText);
        scrollView = findViewById(R.id.scrollView);
        sendButton = findViewById(R.id.sendButton);

        sendButton.setEnabled(false);
        chatText.append("جاري تحميل النموذج...\n\n");

        // تحميل النموذج في thread خلفي باش الواجهة ما تتجمدش
        new Thread(() -> {
            try {
                model = new SalahModel(getApplicationContext());
                handler.post(() -> {
                    chatText.append("النموذج جاهز ✓\n\n");
                    sendButton.setEnabled(true);
                });
            } catch (Throwable e) {
                handler.post(() -> chatText.append("خطأ في تحميل النموذج: " + e + "\n\n"));
            }
        }).start();

        sendButton.setOnClickListener(v -> onSend());
    }

    private void onSend() {
        final String msg = inputText.getText().toString().trim();
        if (msg.isEmpty() || model == null) return;

        chatText.append("أنت: " + msg + "\n\n");
        inputText.setText("");
        sendButton.setEnabled(false);
        scrollToBottom();

        final SalahModel currentModel = model;
        new Thread(() -> {
            String reply;
            try {
                reply = currentModel.generate(msg);
                if (reply.isEmpty()) reply = "(ما تولدش جواب)";
            } catch (Throwable e) {
                reply = "خطأ أثناء التوليد: " + e;
            }
            final String finalReply = reply;
            handler.post(() -> {
                chatText.append("SALAH: " + finalReply + "\n\n");
                sendButton.setEnabled(true);
                scrollToBottom();
            });
        }).start();
    }

    private void scrollToBottom() {
        scrollView.post(() -> scrollView.fullScroll(View.FOCUS_DOWN));
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (model != null) model.close();
    }
}
