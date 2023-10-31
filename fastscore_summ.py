import argparse as ap
import torch
import transformers as tr
import hnswlib
import sentence_transformers as st
import datasets as ds
import numpy as np
import pandas as pd
import os
from model_dict import load_from_catalogue
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import iso639
from auto_gptq import exllama_set_max_input_length


template_icl: str = """
### Instruction:
Score the summarization with respect to the summarized document
on a continuous scale from 0 to 100, where a score of zero means
"irrelevant, factually incorrect and not readable" and score of one hundred means
"relevant, factually correct, good readability".
Source text: "{src}" 
Summary: "{hyp}"
### Response:
Score (0-100):{score}
"""
template_icl = template_icl.strip()


def load_target_dataset(path: str):
        target_dataset = ds.Dataset.from_pandas(pd.read_csv(path, sep='\t'))
        target_dataset = target_dataset.rename_columns({
            'SRC': 'src',
            'TGT': 'hyp'
        })
        return target_dataset

def format_icl(examples, analyzed):
        filled = [template_icl.format(**ex) for ex in examples]
        analyzed['score'] = ''
        filled.append(
            template_icl.format(**analyzed)
        )
        filled = "\n\n".join(filled)
        return filled.strip()

def get_examples(
    input_src: str, 
    index: hnswlib.Index, 
    st_model: st.SentenceTransformer, 
    dataset: ds.Dataset, 
    top_k: int = 3
):
    emb = st_model.encode([f"passage: {input_src}"])[0]
    indexes = index.knn_query(emb, k=top_k+1)[0].squeeze().tolist()[1:]
    objects = []
    for idx in indexes:
        obj = dataset.iloc[idx].to_dict()
        objects.append({
            'src': obj['src'],
            'hyp': obj['hyp'],
            'score': int(round(obj['score']*20))
        })
    return objects

def save_prompts(prompts: list[str], save_path: str):
    with open(save_path, 'w') as f:
        for prompt in prompts:
            f.write(prompt.encode('unicode_escape').decode("utf-8") + '\n')

def main(
    dataset_path: str,
    model_name: str,
    output_path: str,
    output_prompts: str,
    index_cache: str = '.summ.index.bin',
    do_debug: bool = False
):
    
    dataset = ds.load_dataset("Rexhaif/summeval", split='train')
    dataset = dataset.to_pandas()
    dataset = dataset.drop_duplicates(subset=['src'])
    print(f"Loaded {len(dataset)} examples")
    index = hnswlib.Index(space='cosine', dim=768)
    st_model = st.SentenceTransformer("intfloat/e5-base-v2")

    if os.path.exists(index_cache):
        print("Loading index from cache")
        index.load_index(index_cache)
    else:
        print("Building index")
        src_embeddings = st_model.encode([f"passage: {x}" for x in dataset['src'].tolist()], batch_size=512, show_progress_bar=True)
        index.init_index(max_elements=len(src_embeddings), ef_construction=200, M=16)
        print("Adding items to index")
        index.add_items(src_embeddings)
        index.save_index(index_cache)
        print("Index built and saved to cache")

    def format_fn(example):
        return {
            'prompt': format_icl(
                examples=get_examples(input_src=example['src'], top_k=5, index=index, st_model=st_model, dataset=dataset),
                analyzed=example
            )
        }
    
    print("Loading target dataset")
    target_dataset = load_target_dataset(dataset_path)
    target_dataset = target_dataset.map(format_fn, batched=False)

    print("Loading model")
    model, tokenizer, _, _ = load_from_catalogue(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    try:
        model = exllama_set_max_input_length(model, 8192)
    except Exception as e:
        print("Failed to set max input length, model might not support it")
        print(e)
    tokenized_dataset = target_dataset.map(
         lambda x: tokenizer(
            x['prompt'], 
            max_length=8189, 
            truncation=True
        ),
        remove_columns=target_dataset.column_names
    )
    print("Tokenized dataset")
    if do_debug:
        tokenized_dataset = tokenized_dataset.select(range(100))
    collator = DataCollatorWithPadding(tokenizer, padding='longest', max_length=8189, return_tensors='pt')
    dl = DataLoader(tokenized_dataset, batch_size=1, shuffle=False, collate_fn=collator)

    results = []
    print("Generating scores")
    for batch in tqdm(dl):
        batch = {k:v.to(model.device) for k, v in batch.items()}
        outputs = model.generate(**batch, max_new_tokens=3)
        scores = tokenizer.batch_decode(outputs[:, -3:], skip_special_tokens=True)
        print(f"Scores: {scores}")
        for sc in scores:
            if sc.startswith("):"):
                sc = sc[-1]
            try:
                sc = int(sc)
            except:
                sc = 0
                
            results.append(sc)

    print("Saving scores")
    df = pd.DataFrame({
        'score': results
    })
    print(f"{df['score'].mean()} +- {df['score'].std()} [{df['score'].min()}, {df['score'].max()}]")
    df['score'].to_csv(output_path, index=False, header=False)
    print("Saving prompts")
    save_prompts(target_dataset['prompt'], output_prompts)

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='TheBloke/Platypus2-70B-Instruct-GPTQ')
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--output-prompts', type=str, required=True)
    parser.add_argument('--index-cache', type=str, default='.summ.index.bin')
    parser.add_argument('--do-debug', action='store_true', default=False)
    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        output_path=args.output_path,
        output_prompts=args.output_prompts,
        index_cache=args.index_cache,
        do_debug=args.do_debug
    )