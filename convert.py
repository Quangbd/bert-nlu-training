from models import JointTinyBert2
from transformers import BertTokenizer
import torch
import time
import numpy as np
from utils import init_logger, get_intent_labels, get_slot_labels, compute_metrics
import argparse
import os
import tensorflow as tf
from data import load_and_cache_examples


def get_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1234, help='Random seed for initialization')
    parser.add_argument('--no_cuda', action='store_true', help='Avoid using CUDA when available')
    parser.add_argument('--bert_path', type=str, default='data/models/bert-base-uncased',
                        help='Path to save, load model')


    # data
    parser.add_argument('--task', default='snips', type=str, help='The name of the task to train')
    parser.add_argument('--data_dir', default='./data', type=str, help='The input data dir')
    parser.add_argument("--model_dir", default='./data/models/snips_teacher', type=str, help="Path to save, load model")
    parser.add_argument("--output_dir", default='./data/models/snips_student_2', type=str, help="Path to save, load model")
    parser.add_argument("--teacher_model",default='./data/models/bert-base-uncased',type=str,help="The teacher model dir.")
    parser.add_argument("--student_model",default='./data/models/snips_student', type=str, help="The student model dir.")
    parser.add_argument("--tinybert", default='./data/models/tinybert', type=str,help="The tiny model dir.")

    parser.add_argument('--ignore_index', default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')
    parser.add_argument('--max_seq_len', default=50, type=int,
                        help='The maximum total input sequence length after tokenization.')

    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=15.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument('--intent_label_file', default='intent_label.txt', type=str, help='Intent Label file')
    parser.add_argument('--slot_label_file', default='slot_label.txt', type=str, help='Slot Label file')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--stage", default='2.1', help="stage 2.1 or 2.2")
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, help="Whether to run eval on the test set.")
    parser.add_argument("--pred_distill", action='store_true', help="Whether to distill prediction")
    parser.add_argument('--eval_step',type=int,default=50)
    parser.add_argument('--temperature',type=float,default=1.)


    return parser


def convert_examples_to_features(text, max_seq_length, tokenizer):
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:max_seq_length-2]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    input_mask = torch.tensor([input_mask],dtype=torch.long)
    segment_ids = torch.tensor([segment_ids],dtype=torch.long)

    return input_ids,input_mask,segment_ids


if __name__ == '__main__':
    # init_logger()
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("--input_file", default='data/snips/test/seq.in', type=str, help="Input file for prediction")
    # parser.add_argument("--input_query", default=None, type=str, help="Input query for prediction")
    # parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
    # parser.add_argument("--model_dir", default="./data/models/snips_student_2", type=str, help="Path to save, load model")
    #
    # parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    # parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    #
    # pred_config = parser.parse_args()
    #
    # args = get_args(pred_config)
    # device = get_device(pred_config)
    # model = load_model(pred_config, args, device)
    #
    tokenizer = BertTokenizer.from_pretrained('data/models/bert-base-uncased', do_lower_case=True)
    # input_ids,input_mask,segment_ids = convert_examples_to_features('make a reservation at a bakery that has acquacotta in central african republic for five',50,tokenizer)
    #
    # model_onnx_path = "data/models/snips_convert/model2.onnx"
    # dummy_input = (input_ids,segment_ids,input_mask)
    # input_names = ["input_ids",'token_type_ids','attention_mask']
    # output_names = ["output"]
    # torch.onnx.export(model, dummy_input, model_onnx_path,
    #                   input_names=input_names,output_names=['output','output2'], verbose=False,export_params=True)

    # pip install git+https://github.com/onnx/onnx-tensorflow.git
    # onnx-tf convert -i "model2.onnx" -o  "saved_model"
    #
    # TF_PATH = "./my_tf_model2.pb" # where the forzen graph is stored
    # TFLITE_PATH = "data/models/snips_convert/model.tflite"
    # # protopuf needs your virtual environment to be explictly exported in the path
    #
    # # make a converter object from the saved tensorflow file
    # converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model('data/models/snips_convert/saved_model',  # TensorFlow freezegraph .pb model file
    #                                                                input_arrays=["serving_default_input_ids",'serving_default_token_type_ids','serving_default_attention_mask'],
    #                                                                output_arrays=['PartitionedCall','PartitionedCall:1']
    #                                                                  # name of output arrays defined in torch.onnx.export function before.
    #                                                                 )
    # converter.allow_custom_ops = True
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tf_lite_model = converter.convert()
    # # Save the model.
    # with open(TFLITE_PATH, 'wb') as f:
    #     f.write(tf_lite_model)

    parser = get_parse()
    args = parser.parse_args([])
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    eval_sampler = SequentialSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    interpreter = tf.compat.v1.lite.Interpreter(model_path='data/models/snips_convert/model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    eval_loss = 0
    nb_eval_steps = 0
    intent_preds = None
    slot_preds = None
    out_slot_labels_ids=None
    out_intent_label_ids=None

    for batch_ in eval_dataloader:
        input_ids, input_mask, segment_ids, intent_label_ids, slot_labels_ids = batch_

        for i in range(len(input_ids)):
            input_id = input_ids[i,:].unsqueeze(0).detach().cpu().numpy()
            input_m = input_mask[i,:].unsqueeze(0).detach().cpu().numpy()
            segment_id = segment_ids[i,:].unsqueeze(0).detach().cpu().numpy()
            interpreter.set_tensor(input_details[1]['index'], input_id)
            interpreter.set_tensor(input_details[2]['index'], segment_id)
            interpreter.set_tensor(input_details[0]['index'], input_m)
            interpreter.invoke()
            intent_logits = interpreter.get_tensor(output_details[0]['index'])
            slot_logits = interpreter.get_tensor(output_details[1]['index'])

            if intent_preds is None:
                intent_preds = intent_logits
            else:
                intent_preds = np.append(intent_preds, intent_logits, axis=0)

            if slot_preds is None:
                slot_preds = slot_logits
            else:
                slot_preds = np.append(slot_preds, slot_logits, axis=0)
        if out_intent_label_ids is None:
            out_intent_label_ids = intent_label_ids.detach().cpu().numpy()
        else:
            out_intent_label_ids = np.append(
                out_intent_label_ids, intent_label_ids.detach().cpu().numpy(), axis=0)
        if out_slot_labels_ids is None:
            out_slot_labels_ids = slot_labels_ids.detach().cpu().numpy()
        else:
            out_slot_labels_ids = np.append(out_slot_labels_ids, slot_labels_ids.detach().cpu().numpy(), axis=0)

    intent_preds = np.argmax(intent_preds, axis=1)
    slot_preds = np.argmax(slot_preds, axis=2)
    slot_label_lst = get_slot_labels(args)
    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
    slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

    for i in range(out_slot_labels_ids.shape[0]):
        for j in range(out_slot_labels_ids.shape[1]):
            if out_slot_labels_ids[i, j] != 0:
                out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
    print(total_result)
