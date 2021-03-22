# BERT NLU Training

## Training

### Teacher model

```bash
python train.py --teacher_model ./data/models/bert-base-uncased \
                --data_dir ./data \
                --output_dir ./data/models/snips_teacher \
                --max_seq_len 50 \
                --train_batch_size 32 \
                --num_train_epochs 20 \
                --stage 2.0
```

### Intermediate layer distillation
```bash
python train.py --teacher_model ./data/models/bert-base-uncased \
                --data_dir ./data \
                --tinybert ./data/models/tinybert \
                --output_dir ./data/models/snips_teacher_tmp \
                --max_seq_len 50 \
                --train_batch_size 32 \
                --num_train_epochs 100 \
                --stage 2.1
```