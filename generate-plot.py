from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer

# load chrained model
model = GPT2LMHeadModel.from_pretrained('/home/cedric/Downloads/text-generation-1/text-generation')

# load chokenizer
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
)

