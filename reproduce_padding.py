from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

models = [
    "facebook/bart-base",

    "t5-small",
    # "google/mt5-small",
    # "facebook/m2m100_418M",
    # "facebook/wmt19-ru-en",
    # "facebook/blenderbot-400M-distill",
    # "google/bigbird-pegasus-large-arxiv",
    # "allenai/led-base-16384",
    # "microsoft/prophetnet-large-uncased"
]

for model_name in models:
    







for model_name in models:

    # load the seq2seq model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    # sample sentence
    sample_sentence = "generate some numbers"
    encodings = tokenizer(sample_sentence,
                          padding="max_length",
                          max_length=10,
                          return_tensors="pt",
                          return_attention_mask=True,
                          truncation=True)
    print('decoder_start_token_id',model.config.decoder_start_token_id)
    print('decoder_start_token',tokenizer.decode(model.config.decoder_start_token_id))
    print('pad_token_id',tokenizer.pad_token_id)
    print('pad_token',tokenizer.decode(tokenizer.pad_token_id))

    sample_sentence = "generate some numbers"
    encodings = tokenizer(sample_sentence,
                          return_tensors="pt",
                          return_attention_mask=True,
                          truncation=True)
    print('decoder_start_token_id',model.config.decoder_start_token_id)
    print('decoder_start_token',tokenizer.decode(model.config.decoder_start_token_id))
    print('pad_token_id',tokenizer.pad_token_id)
    print('pad_token',tokenizer.decode(tokenizer.pad_token_id))



    # decoder input ids (with a default start token for the model)
    decoder_input_ids = torch.ones(1, 1, dtype=torch.int32) * model.config.decoder_start_token_id
    print('input_ids',encodings.input_ids)
    print('decoder_input_ids',decoder_input_ids)
    # model's forward without any padding for decoder_input_ids (hence without decoder_attn mask)
    outputs = model.forward(input_ids=encodings.input_ids,
                            attention_mask=encodings.attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            return_dict=True)
    next_token_logits = outputs["logits"][:, -1, :]

    # same decoder input ids but padded  + decoder attention mask
    decoder_input_ids_with_padding = torch.ones(1, 3, dtype=torch.int32) * tokenizer.pad_token_id
    decoder_input_ids_with_padding[:, -1] = model.config.decoder_start_token_id
    decoder_attn_mask = torch.zeros(1, 3)
    decoder_attn_mask[:, -1] = 1

    print(encodings.input_ids)
    # model's forward with padding for decoder_input_ids (hence with decoder_attn mask)
    print('input_ids',encodings.input_ids)
    print('decoder_input_ids',decoder_input_ids_with_padding)
    outputs_with_padding = model.forward(input_ids=encodings.input_ids,
                                         attention_mask=encodings.attention_mask,
                                         decoder_input_ids=decoder_input_ids_with_padding,
                                         decoder_attention_mask=decoder_attn_mask,
                                         return_dict=True)
    next_token_logits_with_padding = outputs_with_padding["logits"][:, -1, :]

    # check if padding affects the logits
    if torch.allclose(next_token_logits, next_token_logits_with_padding, atol=1e-3):
        print(f"No issues with model: {model_name}")
    else:
        print(f"Issues with model: {model_name}")