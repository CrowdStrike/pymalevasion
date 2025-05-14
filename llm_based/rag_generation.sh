python compute_sims.py \
    --metadata-path corpus_metadata.csv \
    --scripts-path scripts/ \
    --output-path rankings_test_dirty_top5_similarity.npy

python rag.py \
    --model gpt-4o \
    --scripts-path scripts/ \
    --rankings-path rankings_test_dirty_top5_similarity.npy \
    --system-prompt-path templates/constrained_generation.j2
