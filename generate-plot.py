'''Text generation script from a pretrained GPT2 model.

Usage:
    python generate-plot.py <config>.yaml

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
    model = GPT2LMHeadModel.from_pretrained('/home/cedric/Downloads/text-generation-1/text-generation')

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/home/cedric/Downloads/text-generation-1/text-generation')

    # use transformers's text generation pipeline
    story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    # input prompt 
    input_prompt = '<BOS> ' # dagdag pa ung results ng flames

    # generate text
    text = story_generator(
        input_prompt,
        max_length=100,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        num_return_sequences=5,
    ) # text is a list
