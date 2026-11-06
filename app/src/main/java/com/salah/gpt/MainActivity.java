package com.salah.gpt;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ScrollView;
import android.widget.TextView;

public class MainActivity extends Activity {

    private TextView chatText;
    private EditText inputText;
    private ScrollView scrollView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        chatText = findViewById(R.id.chatText);
        inputText = findViewById(R.id.inputText);
        scrollView = findViewById(R.id.scrollView);
        Button sendButton = findViewById(R.id.sendButton);

        sendButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String msg = inputText.getText().toString().trim();
                if (msg.isEmpty()) return;

                chatText.append("أنت: " + msg + "\n\n");
                // الرد الحقيقي للنموذج غادي يتزاد في الخطوة الجاية
                chatText.append("SALAH: (النموذج ما تربطش بعد)\n\n");
                inputText.setText("");
                scrollView.post(new Runnable() {
                    @Override
                    public void run() {
                        scrollView.fullScroll(View.FOCUS_DOWN);
                    }
                });
            }
        });
    }
                  }
