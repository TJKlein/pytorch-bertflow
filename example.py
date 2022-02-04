from tflow_utils import TransformerGlow, AdamWeightDecayOptimizer
from transformers import AutoTokenizer
import argparse
import os
import wandb
import datasets

parser.add_argument('--model_name_or_path', required=True, type=str, help='Mode for filename to Huggingface model')
parser.add_argument('--train_file', type=str, help='Input training file (text)')
parser.add_argument('--output_dir', required=True, type=str, help='Path where result should be stored')
parser.add_argument('--preprocessing_num_workers', type=int, default=1, help='Number of worker threads for processing data')
parser.add_argument('--overwrite_cache', type=bool, defaule=False, help='Indicate whether cached features should be overwritten')
parser.add_argument('--pad_to_max_length', type=bool, default=False, help='Indicate whether tokens sequence should be padded')
parser.add_argument('--max_seq_length', type=int, default=32, help='Input max sequence length in tokens')
parser.add_argument('--description', type=str, help='Experiment description')
parser.add_argument('--tags', type=str, help='Annotation tags for wandb, comma separated')


parser = argparse.ArgumentParser()


if __name__ == '__main__':
    
    FLAGS = parser.parse_args()
    
    
    
    if not (FLAGS.tags is None):
        FLAGS.tags = [item for item in FLAGS.tags.split(',')]

    if not(FLAGS.tags is None):
        wandb.init(project=FLAGS.description, tags=data_args.tags)

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
    
    bertflow = TransformerGlow(FLAGS.model_name_or_path, pooling='first-last-avg')  # pooling could be 'mean', 'max', 'cls' or 'first-last-avg' (mean pooling over the first and the last layers)
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_name_or_path)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters= [
        {
            "params": [p for n, p in bertflow.glow.named_parameters()  \
                            if not any(nd in n for nd in no_decay)],  # Note only the parameters within bertflow.glow will be updated and the Transformer will be freezed during training.
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in bertflow.glow.named_parameters()  \
                            if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamWeightDecayOptimizer(
        params=optimizer_grouped_parameters, 
        lr=1e-3, 
        eps=1e-6,
    )

    extension = os.path.splitext(FLAGS.train_file)[1]
    datasets = load_dataset(extension, data_files=FLAGS.train_file, cache_dir="./data/")
    column_names = datasets["train"].column_names

    
     # Unsupervised datasets
    sent0_cname = column_names[0]
    
    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the 
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[sent0_cname])

        # Avoid "None" fields 
        for idx in range(total):
            if examples[sent0_cname][idx] is None:
                examples[sent0_cname][idx] = " "
          
        sentences = examples[sent0_cname]

        
        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if FLAGS.pad_to_max_length else False,
        )

      
            
        return sent_features
    
    
    train_dataset = datasets["train"].map(
            prepare_features,
            batched=True,
            num_proc=FLAGS.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not FLAGS.overwrite_cache,
        )
    
    
    
    # Important: Remember to shuffle your training data!!! This makes a huge difference!!!
    sentences = ['This is sentence A.', 'And this is sentence B.']  # Please replace this with your datasets (single sentences).
    model_inputs = tokenizer(
        sentences,
        add_special_tokens=True,
        return_tensors='pt',
        max_length=512,
        padding='longest',
        truncation=True
    )
    bertflow.train()
    z, loss = bertflow(model_inputs['input_ids'], model_inputs['attention_mask'], return_loss=True)  # Here z is the sentence embedding
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    bertflow.save_pretrained('output')  # Save model
    bertflow = TransformerGlow.from_pretrained('output')  # Load model