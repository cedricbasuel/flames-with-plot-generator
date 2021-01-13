from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    GPT2LMHeadModel,
    AutoTokenizer,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    # LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoModelWithLMHead,
)
import pandas



# tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
print('loading tokenizer...')
# tokenizer = GPT2Tokenizer.from_pretrained(
#     'gpt2', 
#     bos_token='<BOS>', 
#     eos_token='<EOS>', 
#     pad_token='<|pad|>'
#     )


# model
print('loading pretrained GPT2 model...')
# model = GPT2LMHeadModel.from_pretrained('gpt2')
model = AutoModelWithLMHead.from_pretrained('gpt2')

# add special token dictio
special_tokens_dict = {
    'bos_token': '<BOS>',
    'eos_token': '<EOS>',
}

# model resize token emb??? AHHH kasi nga nag dagdag ng special tokens so mag-aadjust din ung length ng embeddings
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))


# load data
# data is .txt file with one entry per line
print('loading data...')
train_data = TextDataset(
    tokenizer=tokenizer, 
    file_path='/home/cedric/Desktop/cv-tf2/plots_train_gpt2.txt',
    block_size=50
    )

test_data = TextDataset(
    tokenizer=tokenizer,
    file_path='/home/cedric/Desktop/cv-tf2/plots_test_gpt2.txt',
    block_size=50
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# train model
print('training...')
training_args = TrainingArguments(
    output_dir="/home/cedric/Desktop/cv-tf2/text-generation",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    # evaluate_during_training=False,
    # logging_steps=500,
    # per_device_train_batch_size=4,
    num_train_epochs=3,
    # save_total_limit=1,
    save_steps=10000,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator
)

trainer.train()

# save model
print('saving model...')
trainer.save_model('/home/cedric/Desktop/cv-tf2/text-generation')
tokenizer.save_pretrained('/home/cedric/Desktop/cv-tf2/text-generation')

