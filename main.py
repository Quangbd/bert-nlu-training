import argparse

from trainer import Trainer
from utils import init_logger, set_seed
from data import load_and_cache_examples
from transformers import BertTokenizer


def main(args):
    init_logger()
    set_seed(args)
    if args.stage == '2.1':
        tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=True)
    elif args.stage == '2.2':
        tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=True)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset, dev_dataset,test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == '__main__':
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
    parser.add_argument("--pred_distill", action='store_true', default=True, help="Whether to distill prediction")
    parser.add_argument('--eval_step',type=int,default=50)
    parser.add_argument('--temperature',type=float,default=1.)

    main(parser.parse_args())
