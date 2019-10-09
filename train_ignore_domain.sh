PYTHONPATH=./ CUDA_VISIBLE_DEVICES=1 ./scripts/gpt-baseline train ./pp-metalwoz-dir/metalwoz-v1-normed.zip \
    --preproc-dir ./pp-metalwoz-dir \
    --output-dir  ./metalwoz-retrieval-model_ignore_domain \
    --eval-domain dialogues/EVENT_RESERVE.txt \
    --test-domain dialogues/EVENT_RESERVE.txt \
    --method ignore-domain

# eval-dom and test-dom is fake!! becareful
# you choose check in gpt-trainer (variable)  eval_domains
