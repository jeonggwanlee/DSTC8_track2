./gpt-baseline train ./pp-metalwoz-dir/metalwoz-v1-normed.zip \
    --preproc-dir ./pp-metalwoz-dir \
    --output-dir  ./metalwoz-retrieval-model_sh \
    --eval-domain dialogues/ALARM_SET.txt --test-domain dialogues/EVENT_RESERVE.txt
