import argparse

from gector.gec_model import GecBERTModel
from tqdm import tqdm

from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field, ArrayField
from allennlp.data.instance import Instance
from gector.tokenizer_indexer import PretrainedBertIndexer
import numpy as np
from utils.helpers import SEQ_DELIMETERS, START_TOKEN
from typing import Dict, List
from random import random

def get_token_indexers(model_name, max_pieces_per_token=5, lowercase_tokens=True, special_tokens_fix=0):
    bert_token_indexer = PretrainedBertIndexer(
        pretrained_model=model_name,
        max_pieces_per_token=max_pieces_per_token,
        do_lowercase=lowercase_tokens,
        special_tokens_fix=special_tokens_fix
    )
    return {'bert': bert_token_indexer}

class gectorReader():
    def __init__(self, token_indexer, max_len) -> None:
        self._delimeters = SEQ_DELIMETERS
        self._token_indexers = token_indexer
        self._max_len = max_len
    
    def extract_tags(self, tags: List[str]):
        op_del = self._delimeters['operations']

        labels = [x.split(op_del) for x in tags]

        labels = [x[0] for x in labels]

        detect_tags = ["CORRECT" if label == "$KEEP" else "INCORRECT" for label in labels]
        return labels, detect_tags

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None,
                         words: List[str] = None,) -> Instance:  # type: ignore
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": words})
        if tags is not None:
            labels, detect_tags = self.extract_tags(tags)

            fields["labels"] = SequenceLabelField(labels, sequence,
                                                  label_namespace="labels")
            fields["d_tags"] = SequenceLabelField(detect_tags, sequence,
                                                  label_namespace="d_tags")
        return Instance(fields)

    def read(self, file_path):
        with open(file_path, "r") as data_file:
            print("Reading instances from lines in file at: %s", file_path)
            for i, line in enumerate(data_file):
                line = line.strip("\n")
                # skip blank and broken lines
                if not line:
                    raise ValueError('line is empty')

                tokens_and_tags = [pair.rsplit(self._delimeters['labels'], 1)
                                    for pair in line.split(self._delimeters['tokens'])]
                try:
                    tokens = [Token(token) for token, tag in tokens_and_tags]
                    tags = [tag for token, tag in tokens_and_tags]
                except ValueError:
                    tokens = [Token(token[0]) for token in tokens_and_tags]
                    tags = None
                if tokens and tokens[0] != Token(START_TOKEN):
                    tokens = [Token(START_TOKEN)] + tokens

                words = [x.text for x in tokens]
                if self._max_len is not None:
                    tokens = tokens[:self._max_len]
                    tags = None if tags is None else tags[:self._max_len]
                instance = self.text_to_instance(tokens, tags, words)
                if instance:
                    yield instance

def predict_for_file(input_data, output_file, model, batch_size=32, to_normalize=False):
    predictions = []
    batch = []
    for instance in tqdm(input_data):
        if not instance:
            raise ValueError('instance is None')
        batch.append(instance)
        if len(batch) == batch_size:
            correct_probs = model.handle_batch_for_token_correct_probs(batch)
            assert len(correct_probs) == batch_size, len(correct_probs)
            predictions.extend(correct_probs)

            batch = []
    if batch:
        correct_probs = model.handle_batch_for_token_correct_probs(batch)
        predictions.extend(correct_probs)

    np.savez(output_file, *predictions)
    print(predictions[:3])
    return len(predictions)

def predict_for_entropy(input_data, output_file, model, batch_size=32, to_normalize=False):
    predictions = []
    batch = []
    for instance in tqdm(input_data):
        if not instance:
            raise ValueError('instance is None')
        batch.append(instance)
        if len(batch) == batch_size:
            entropy = model.handle_batch_for_token_entropy(batch)
            assert len(entropy) == batch_size, len(entropy)
            predictions.extend(entropy)

            batch = []
    if batch:
        entropy = model.handle_batch_for_token_entropy(batch)
        predictions.extend(entropy)

    np.savez(output_file, *predictions)
    print(predictions[:3])
    return len(predictions)

def predict_for_quality(input_data, entropy_output_file, origin_output_file, former_model, latter_model, batch_size=32, to_normalize=False):
    all_delta_ppl = []
    lengths = []
    batch = []
    for instance in tqdm(input_data):
        if not instance:
            raise ValueError('instance is None')
        batch.append(instance)
        if len(batch) == batch_size:
            former_ppl, lens = former_model.handle_batch_for_ppl(batch)
            latter_ppl, lens = latter_model.handle_batch_for_ppl(batch)
            assert len(former_ppl) == batch_size, f"{len(former_ppl)}, {batch_size}"
            all_delta_ppl.extend([f -l for f,l  in zip(former_ppl, latter_ppl)])
            lengths.extend(lens)
            batch = []
    if batch:
        former_ppl, lens = former_model.handle_batch_for_ppl(batch)
        latter_ppl, lens = latter_model.handle_batch_for_ppl(batch)
        all_delta_ppl.extend([f -l for f,l in zip(former_ppl, latter_ppl)])
        lengths.extend(lens)

    def save(origin_array_list, lengths, file):
        dtype = np.array(origin_array_list[0]).dtype
        return_array = np.asarray(np.zeros((len(lengths), max(lengths)), dtype=dtype) , dtype=dtype)
        for i, _len in enumerate(lengths):
            slices = tuple([i, slice(0, _len)])
            return_array[slices] = origin_array_list[i]
        np.savez(file, data=return_array, lengths=np.array(lengths))
    
    print(all_delta_ppl[:100])
    save(all_delta_ppl, lengths, origin_output_file)

    threshold = np.percentile(all_delta_ppl, 10)
    all_weights = [1 if d < threshold else 0 for d in all_delta_ppl]
    print(all_weights[:100])
    save(all_weights, lengths, entropy_output_file)
    return len(all_weights)

def main(args):
    # get all paths
    from utils.helpers import get_weights_name
    token_indexer = get_token_indexers(model_name=get_weights_name(args.transformer_model, args.lowercase_tokens),
                                        max_pieces_per_token=args.pieces_per_token,
                                        lowercase_tokens=args.lowercase_tokens,
                                        special_tokens_fix=args.special_tokens_fix,
                                        )
    reader = gectorReader(token_indexer, args.max_len)
    former_model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.former_model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         del_confidence=args.additional_del_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights)
    
    latter_model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=args.latter_model_path,
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=args.min_error_probability,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=args.transformer_model,
                         special_tokens_fix=args.special_tokens_fix,
                         log=False,
                         confidence=args.additional_confidence,
                         del_confidence=args.additional_del_confidence,
                         is_ensemble=args.is_ensemble,
                         weigths=args.weights)

    # if args.generate_entropy:
    #     cnt_corrections = predict_for_entropy(reader.read(args.input_file), args.output_file, model,
    #                                     batch_size=args.batch_size, 
    #                                     to_normalize=args.normalize)
    # else:
    #     cnt_corrections = predict_for_file(reader.read(args.input_file), args.output_file, model,
    #                                    batch_size=args.batch_size, 
    #                                    to_normalize=args.normalize)
    cnt_corrections = predict_for_quality(reader.read(args.input_file), args.entropy_output_file, args.entropy_output_file + 'origin' , former_model, latter_model,
                                       batch_size=args.batch_size, 
                                       to_normalize=args.normalize)
    # evaluate with m2 or ERRANT
    print(f"Produced overall lines: {cnt_corrections}")


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--former_model_path',
                        help='Path to the model file.', nargs='+',
                        required=True)
    parser.add_argument('--latter_model_path',
                        help='Path to the model file.', nargs='+',
                        required=True)
    parser.add_argument('--vocab_path',
                        help='Path to the model file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--input_file',
                        help='Path to the evalset file',
                        required=True)
    parser.add_argument('--entropy_output_file',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all longer will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--transformer_model',
                        choices=['bert', 'gpt2', 'transformerxl', 'xlnet', 'distilbert', 'roberta', 'albert'
                                 'bert-large', 'roberta-large', 'xlnet-large'],
                        help='Name of the transformer model.',
                        default='roberta')
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_confidence',
                        type=float,
                        help='How many probability to add to $KEEP token.',
                        default=0)
    parser.add_argument('--additional_del_confidence',
                        type=float,
                        help='How many probability to add to $DELETE token.',
                        default=0)
    parser.add_argument('--min_error_probability',
                        type=float,
                        help='Minimum probability for each action to apply. '
                             'Also, minimum error probability, as described in the paper.',
                        default=0.0)
    parser.add_argument('--special_tokens_fix',
                        type=int,
                        help='Whether to fix problem with [CLS], [SEP] tokens tokenization. '
                             'For reproducing reported results it should be 0 for BERT/XLNet and 1 for RoBERTa.',
                        default=1)
    parser.add_argument('--is_ensemble',
                        type=int,
                        help='Whether to do ensembling.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--normalize',
                        help='Use for text simplification.',
                        action='store_true')
    parser.add_argument('--pieces_per_token',
                        type=int,
                        help='The max number for pieces per token.',
                        default=5)
    parser.add_argument('--generate_entropy',
                        action='store_true')
    
    args = parser.parse_args()
    main(args)

"""

CUDA_VISIBLE_DEVICES=3 nohup python /home/ljh/GEC/gector/prepare_data.py \
--former_model_path /home/ljh/GEC/gector/output/baseline/BW_stage1_roberta_large/model_state_epoch_3.th \
--latter_model_path /home/ljh/GEC/gector/output/baseline/BW_stage2_roberta_large/model_state_epoch_0.th \
--transformer_model roberta-large \
--vocab_path /home/ljh/GEC/gector/data/output_vocabulary \
--input_file /home/ljh/GEC/gector/data/stage1.train  \
--batch_size 1024 \
--entropy_output_file /home/ljh/GEC/gector/data/stage1.2020_dwt   \
--batch_size 1024 2>&1 >/home/ljh/GEC/gector/bash/adaptive_train/ablation/reimp_2020_data_weighted_training/prepare.log &


"""