import argparse
from transformers import BertTokenizer
from utils import set_seed


def main(args):
    set_seed(args)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=args.bert_path)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    print(encoded_input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1234, help='Random seed for initialization')
    parser.add_argument('--no_cuda', action='store_true', help='Avoid using CUDA when available')
    parser.add_argument('--bert_path', type=str, default='data/models/bert-base-uncased',
                        help='Path to save, load model')

    main(parser.parse_args())
