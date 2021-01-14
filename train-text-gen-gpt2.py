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
import logging

logging.basicConfig(
    format='[TEXT-GENETRATION:TRAIN] %(asctime)s %(message)s', 
    datefmt='%m/%d/%Y %I:%M:%S %p', 
    level=logging.DEBUG
    )

if __name__ == '__main__':

    CONFIG_FILE = sys.argv[1]

    with open(CONFIG_FILE) as cfg:
        config = yaml.safe_load(cfg)

    # tokenizer
    logging.info('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(config['gpt_train']['model_name'])
    # tokenizer = GPT2Tokenizer.from_pretrained(
    #     'gpt2', 
    #     bos_token='<BOS>', 
    #     eos_token='<EOS>', 
    #     pad_token='<|pad|>'
    #     )


    # model
    logging.info('Loading pretrained weights GPT2 model...')
    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = AutoModelWithLMHead.from_pretrained(config['gpt_train']['model_name'])

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
    logging.info('Loading data...')
    train_data = TextDataset(
        tokenizer=tokenizer, 
        file_path=config['gpt_train']['train_data'],
        block_size=config['gpt_train']['block_size']
        )

    test_data = TextDataset(
        tokenizer=tokenizer,
        file_path=config['gpt_train']['eval_data'],
        block_size=config['gpt_train']['block_size']
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # train model
    logging.info('Training model...')
    training_args = TrainingArguments(
        output_dir=config['gpt_train']['dir'],
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        # evaluate_during_training=False,
        # logging_steps=500,
        # per_device_train_batch_size=4,
        num_train_epochs=config['gpt_train']['epochs'],
        # save_total_limit=1,
        save_steps=config['gpt_train']['save_steps'],
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
    logging.info('Saving model...')
    trainer.save_model(config['gpt_train']['dir'])
    tokenizer.save_pretrained(config['gpt_train']['dir'])

