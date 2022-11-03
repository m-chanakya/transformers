
from transformers import AutoTokenizer
code = "public void METHOD_1 ( TYPE_1 VAR_1 ) throws java.lang.Exception { super . METHOD_1 ( VAR_1 ) ; METHOD_2 ( VAR_1 ) ; }"
tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base", language_codes="base")
model_inputs = tokenizer([code])
print(tokenizer.decode(model_inputs['input_ids'][0], skip_special_tokens=True, clean_up_tokenization_spaces=False))
# public void METHOD_1 ( TYPE_1 VAR_1 ) throws .lang.Exception { super . METHOD_1 ( VAR_1 ) ; METHOD_2 ( VAR_1 ) ; }
