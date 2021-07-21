LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 python -m soundscape_generation.train --num_epochs $EPOCHS --batch_size $BATCH_SIZE --model_to_load $MODEL_TO_LOAD
