'''FLAMES game with plot generator.

Usage:
    python flames-with-generate-plot.py config.py

Author:
    Cedric Basuel
'''


import flames
import logging
from functools import wraps
from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer
import random
import sys
import yaml
import torch

logging.basicConfig(
    # filename='train_image.log',
    format='[FLAMES-WITH-TEXTGEN] %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', 
    level=logging.DEBUG
    )

CONFIG_FILE = sys.argv[1]

with open(CONFIG_FILE) as cfg:
    config = yaml.safe_load(cfg)

device = torch.device('cuda')

name1 = input('Enter name: ')
name2 = input('Enter name: ')

unique_letters = flames.remove_common_letters(name1, name2)
flames_status = flames.get_flames_status(unique_letters)
print('FLAMES status:', flames_status)

prompts = {
    'Friendship': [', who is friends with ', ', in a friendship with ', ', a good friend of '],
    'Love': [', who is in love with ', ' who loves ', ', lover of ', ', the partner of '],
    'Affection': [', who has affection for ', ' who is affectionate with ', ' who has nothing but affection for '],
    'Marriage': [', who is married to ', ' the partner of '],
    'Enemy': [', the enemy of ',  ', who hates ', ' the rival of ', ' who is engaged in a rivalry with '],
    'Sibling': [', the sibling of '],
}

# load trained model
model = GPT2LMHeadModel.from_pretrained(config['gpt_generate']['dir'])

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['gpt_generate']['dir'])

# use transformers's text generation pipeline
story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device='cuda:0')

# form input prompt from names and flames_status 
input_prompt = '<BOS> ' + name1 + random.choice(prompts[flames_status]) + name2

# generate text
text = story_generator(
    input_prompt,
    max_length=config['gpt_generate']['max_length'],
    do_sample=config['gpt_generate']['do_sample'],
    temperature=config['gpt_generate']['temperature'],
    top_p=config['gpt_generate']['top_p'],
    top_k=config['gpt_generate']['top_k'],
    repetition_penalty=config['gpt_generate']['rep_penalty'],
    num_return_sequences=config['gpt_generate']['num_return_sequences'],
)

# clean generated text
plots_list = [plot['generated_text'].replace('<BOS> ','') for plot in text]

for plot in plots_list:
    print('*' * 30)
    print(plot)
    
