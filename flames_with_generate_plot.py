'''FLAMES game with plot generator.

Usage:
    python flames-with-generate-plot.py config.yaml

Author:
    Cedric Basuel
'''

# TO DO docstrings
# TO DO [app.py] split flames from generate_plot
# TO DO transfer config (or enuf na tong yaml?)
# TO DO [UI] animations or progress bar

import flames
import logging
from functools import wraps
from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer
import random
import sys
import yaml
import torch
import time

logging.basicConfig(
    # filename='train_image.log',
    format='[FLAMES-WITH-STORY-GENERATOR] %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', 
    level=logging.DEBUG
    )

def timer(func):
    @wraps(func)
    def wrapper_time(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        run_time = time.time() - start_time
        logging.info('{} took {} seconds to finish.'.format(func.__name__, run_time))
        return value
    return wrapper_time

@timer
def load_gpt_model(model_path, tokenizer_path, device):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    story_generator = TextGenerationPipeline(
        model=model, 
        tokenizer=tokenizer, 
        # device=0
        )

    logging.info('GPT2 model loaded.')

    return story_generator

def create_input_prompts(name1, name2, flames_status, n_plots):

    prompts = {
        'Friendship': ['who is friends with', 'in a friendship with', 'a good friend of'],
        'Love': ['who is in love with', 'who loves', 'lover of', 'the partner of'],
        'Affection': ['who has affection for', 'who is affectionate with', 'who has nothing but affection for'],
        'Marriage': ['who is married to', 'the partner of'],
        'Enemy': ['the enemy of',  'who hates', 'the rival of', 'who is engaged in a rivalry with'],
        'Sibling': ['the sibling of'],
    }

    context = prompts[flames_status]

    input_prompts = ['<BOS> ' + name1 + ', ' + random.choice(context) + ' ' + name2 + ' ' for n_plot in range(n_plots)]

    logging.info('Input prompts created.')

    return input_prompts

@timer
def generate_plot(
    story_generator, input_prompts, temperatures, max_length,
    do_sample, top_p, top_k, repetition_penalty, num_return_sequences,
    ):
    
    plots = []

    if len(input_prompts) != len(temperatures):
        raise AssertionError('Length of input prompts and temperatures must be equal to n_plots.')

    else:
        for (in_prompt, temp) in zip(input_prompts, temperatures):
            text = story_generator(
                in_prompt,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
            )
            plots.append(text[0])
    
        logging.info('Plots generated.')

    return plots


if __name__ == "__main__":

    CONFIG_FILE = sys.argv[1]

    with open(CONFIG_FILE) as cfg:
        config = yaml.safe_load(cfg)

    # Get names
    name1 = input('Enter name: ')
    name2 = input('Enter name: ')

    # Get flames status
    unique_letters = flames.remove_common_letters(name1, name2)
    flames_status = flames.get_flames_status(unique_letters)
    print('FLAMES status:', flames_status)

    input_prompts = create_input_prompts(name1, name2, flames_status, n_plots=config['gpt_generate']['n_plots'])

    story_generator = load_gpt_model(
        model_path=config['gpt_generate']['dir'], 
        tokenizer_path=config['gpt_generate']['dir'],
        device=0)

    plots = generate_plot(
        story_generator,
        input_prompts=input_prompts,
        temperatures=config['gpt_generate']['temperatures'],
        max_length=config['gpt_generate']['max_length'],
        do_sample=config['gpt_generate']['do_sample'],
        top_p=config['gpt_generate']['top_p'],
        top_k=config['gpt_generate']['top_k'],
        repetition_penalty=config['gpt_generate']['rep_penalty'],
        num_return_sequences=config['gpt_generate']['num_return_sequences'],
        )

    # Clean generated text
    plots_list = [plot['generated_text'].replace('<BOS> ','') for plot in plots]

    for plot in plots_list:
        print('*' * 30)
        print(plot)
        
