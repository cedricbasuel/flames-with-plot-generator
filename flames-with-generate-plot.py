import flames
import logging
from functools import wraps
from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer


logging.basicConfig(
    # filename='train_image.log',
    format='[TEXT-GENETRATION] %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', 
    level=logging.DEBUG
    )


