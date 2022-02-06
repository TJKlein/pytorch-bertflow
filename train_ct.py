#import senteval
import random
import sys
import argparse
import os
import wandb
import numpy as np
from datasets import load_dataset
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, LoggingHandler, models, util, InputExample
from sentence_transformers import losses
import os
import gzip
import csv
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument("--no_cuda", action='store_true',
                    help="Avoid using CUDA when available")
parser.add_argument('--model_name_or_path', required=True, type=str, help='Mode for filename to Huggingface model')
parser.add_argument('--train_file', type=str, help='Input training file (text)')
parser.add_argument('--output_dir', required=True, type=str, help='Path where result should be stored')
parser.add_argument('--preprocessing_num_workers', type=int, default=1, help='Number of worker threads for processing data')
parser.add_argument('--overwrite_cache', type=bool, default=False, help='Indicate whether cached features should be overwritten')
parser.add_argument('--pad_to_max_length', type=bool, default=True, help='Indicate whether tokens sequence should be padded')
parser.add_argument('--max_seq_length', type=int, default=32, help='Input max sequence length in tokens')
parser.add_argument('--overwrite_output_dir', type=bool, default=True, help="If data in output directory should be overwritten if already existing.")
parser.add_argument('--learning_rate', type=float,
                    default=1e-5, help='SGD learning rate')
parser.add_argument('--num_train_epochs', type=int, default=3,
                    help='Number of trainin epochs')
parser.add_argument('--per_device_train_batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--description', type=str, help='Experiment description')
parser.add_argument('--tags', type=str, help='Annotation tags for wandb, comma separated')
parser.add_argument('--eval_steps', type=int, default=250, help='Frequency of model selection evaluation')
parser.add_argument('--seed', type=int, default=48, help='Random seed for reproducability')
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    
    FLAGS = parser.parse_args()
    
    
    
    if not (FLAGS.tags is None):
        FLAGS.tags = [item for item in FLAGS.tags.split(',')]

    if not(FLAGS.tags is None):
        wandb.init(project=FLAGS.description, tags=FLAGS.tags)

    else:
        wandb.init(project=FLAGS.description)

    wandb.config.update({"Command Line": 'python '+' '.join(sys.argv[0:])})

    if not(wandb.run.name is None):
        output_name = wandb.run.name
    else:
        output_name = 'dummy-run'
    
    FLAGS.output_dir = os.path.join(
        FLAGS.output_dir, output_name)
    
    
    if (
        os.path.exists(FLAGS.output_dir)
        and os.listdir(FLAGS.output_dir)
        and not FLAGS.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({FLAGS.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )


    if FLAGS.local_rank == -1 or FLAGS.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        FLAGS.n_gpu = torch.cuda.device_count()
    
    # make sure we can make all the experiment results reproducible
    set_seed(FLAGS.seed)
    ################# Download and load STSb #################
    data_folder = 'data/stsbenchmark'
    sts_dataset_path = f'{data_folder}/stsbenchmark.tsv.gz'

    if not os.path.exists(sts_dataset_path):
        util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

    datasets = load_dataset(path = os.path.split(os.path.abspath(FLAGS.train_file))[0], data_files={'train': os.path.split(os.path.abspath(FLAGS.train_file))[1]}, cache_dir="./data/")
    

    dev_samples = []
    test_samples = []
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)

    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')
    #test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')

    ################# Intialize an SBERT model #################
    word_embedding_model = models.Transformer(FLAGS.model_name_or_path, max_seq_length=FLAGS.max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    pos_neg_ratio = 8 
    # For ContrastiveTension we need a special data loader to construct batches with the desired properties
    train_dataloader =  losses.ContrastiveTensionDataLoader(datasets["train"]['text'], batch_size=FLAGS.per_device_train_batch_size, pos_neg_ratio=pos_neg_ratio)

    # As loss, we losses.ContrastiveTensionLoss
    train_loss = losses.ContrastiveTensionLoss(model)


    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=FLAGS.num_train_epochs,
        evaluation_steps=FLAGS.eval_steps,
        weight_decay=0,
        warmup_steps=0,
        optimizer_class=torch.optim.RMSprop,
        optimizer_params={'lr': FLAGS.learning_rate},
        output_path=FLAGS.output_dir,
        use_amp=False    #Set to True, if your GPU has optimized FP16 cores
    )

        

   
