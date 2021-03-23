# BERT NLU Training

## Training

- Step 1: Download teacher model - fine-tuned Bert-12
  (Ex: [teacher model](https://huggingface.co/bert-base-uncased/tree/main)).
- Step 2: Download tinybert - original TinyBert for weights initialization
  (Ex: [tinybert](https://drive.google.com/uc?export=download&id=1dDigD7QBv1BmE6pWU71pFYPgovvEqOOj)).
- Step 3: use `train.py` to run the teacher model.
```bash
python train.py --teacher_model ./data/models/bert-base-uncased \
                --data_dir ./data \
                --output_dir ./data/models/snips_teacher \
                --max_seq_len 50 \
                --train_batch_size 32 \
                --num_train_epochs 20 \
                --stage 2.0
```
- Step 4: use `train.py` to run the intermediate layer distillation.
```bash
python train.py --teacher_model ./data/models/snips_teacher \
                --data_dir ./data \
                --tinybert ./data/models/tinybert \
                --output_dir ./data/models/snips_student_tmp \
                --max_seq_len 50 \
                --train_batch_size 32 \
                --num_train_epochs 100 \
                --stage 2.1
```
- Step 5: use `train.py` to run the prediction layer distillation.
```bash
python train.py --pred_distill  \
                --teacher_model ./data/models/snips_teacher \
                --student_model ./data/models/snips_student_tmp \
                --data_dir ./data \
                --tinybert ./data/models/tinybert \
                --output_dir ./data/models/snips_student \
                --num_train_epochs  20  \
                --eval_step 50 \
                --max_seq_len 50 \
                --train_batch_size 8 \
                --stage 2.2
```
- Step 6: use `predict.py` to run the prediction layer distillation.
```
# predict with input from file
python predict.py  \
                --input_file ${input_txt_file}$ \
                --output_file ${output_file}$ \
                --model_dir ${OUTPUT_DIR}$ 

# predict with input from file
python predict.py  \
        --input_query "input text" \
        --output_file ${output_file}$ \
        --model_dir ${OUTPUT_DIR}$                 
```