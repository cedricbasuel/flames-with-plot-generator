'''Text generation script from a pretrained GPT2 model.

Usage:
    python generate-plot.py config.yaml

Author:
    Cedric Basuel

'''

import yaml
import time
import logging
from functools import wraps
from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer


logging.basicConfig(
    # filename='train_image.log',
    format='[TEXT-GENETRATION] %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', 
    level=logging.DEBUG
    )


if __name__ == '__main__':

    CONFIG_FILE = sys.argv[1]

    with open(CONFIG_FILE) as cfg:
        config = yaml.safe_load(cfg)

    # load trained model
    model = GPT2LMHeadModel.from_pretrained(config['gpt_generate']['dir'])

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['gpt_generate']['dir'])

    # use transformers's text generation pipeline
    story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    # input prompt 
    input_prompt = '<BOS>'

    # generate text
    text = story_generator(
        input_prompt,
        max_length=config['gpt_generate']['max_length'],
        do_sample=True,
        temperature=config['gpt_generate']['temperature'],
        top_p=config['gpt_generate']['top_p'],
        top_k=config['gpt_generate']['top_k'],
        repetition_penalty=config['gpt_generate']['rep_penalty'],
        num_return_sequences=config['gpt_generate']['num_return_sequences'],
    )

    # clean generated text
    plots_list = [plot['generated_text'].replace('<BOS>','') for plot in text]
