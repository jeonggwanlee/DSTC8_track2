PYTHONPATH=./ CUDA_VISIBLE_DEVICES=1 ./scripts/gpt-baseline train ./pp-metalwoz-dir/metalwoz-v1-normed.zip \
    --preproc-dir ./pp-metalwoz-dir \
    --output-dir  ./metalwoz-retrieval-model_meta_learning_version1 \
    --eval-domain dialogues/ALARM_SET.txt \
    --eval-domain dialogues/AGREEMENT_BOT.txt \
    --eval-domain dialogues/EDIT_PLAYLIST.txt \
    --method meta-learning
