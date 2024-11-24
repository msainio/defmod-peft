import logging
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)

def _fill_prompt_template(
        row, colname_def, colname_ex, colname_word, do_eval, eos_token,
        template):
    if do_eval:
        prompt = template.format(
                example=row[colname_ex],
                word=row[colname_word]
                )
    else:
        prompt = template.format(
                definition=row[colname_def],
                example=row[colname_ex],
                word=row[colname_word]
                )
    return prompt + eos_token

def _get_input_texts(
        colname_def, colname_ex, colname_word, data, do_eval, eos_token,
        data_lang, prompt_templates):
    string_data = data.map(lambda x: str(x))
    task = "generate" if do_eval else "train"
    template = prompt_templates[data_lang][task]
    args = (colname_def, colname_ex, colname_word,
            do_eval, eos_token, template)
    input_texts = string_data.apply(_fill_prompt_template, axis=1, args=args)
    return input_texts.to_list()

def _get_labels(model_inputs, pad_token_id):
    labels = model_inputs["input_ids"].clone().detach()
    labels[labels == pad_token_id] = -100
    return labels

def _get_model_inputs(
        colname_def, colname_ex, colname_word, data, do_eval, max_length,
        data_lang, prompt_templates, tokenizer):
    eos_token = tokenizer.eos_token
    input_texts = _get_input_texts(
            colname_def, colname_ex, colname_word, data, do_eval, eos_token,
            data_lang, prompt_templates)
    model_inputs = tokenizer(
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            text=input_texts,
            truncation=True,
            )
    model_inputs["labels"] = _get_labels(model_inputs, tokenizer.pad_token_id)
    return model_inputs

def _prepare_dataset(
        batch_size, colname_def, colname_ex, colname_word, do_eval, filepath,
        max_length, num_workers, pin_memory, data_lang, prompt_templates,
        shuffle, tokenizer):
    data = pd.read_csv(filepath)
    model_inputs = _get_model_inputs(
            colname_def, colname_ex, colname_word, data, do_eval, max_length,
            data_lang, prompt_templates, tokenizer)
    dataset = TensorDataset(
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs["labels"],
            )
    dataloader = DataLoader(
            batch_size=batch_size,
            dataset=dataset,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle,
            )
    return data, dataloader

def prepare_datasets(
        config, num_workers, pin_memory, prompt_templates, tokenizer):
    kwargs = {
            "batch_size": config["batch_size"],
            "colname_def": config["colname_def"],
            "colname_ex": config["colname_ex"],
            "colname_word": config["colname_word"],
            "do_eval": config["do_eval"],
            "max_length": config["max_length"],
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "data_lang": config["data_lang"],
            "prompt_templates": prompt_templates,
            "tokenizer": tokenizer,
            }
    if config["do_eval"]:
        test_data, test_loader = _prepare_dataset(
                filepath=config["test_file"], shuffle=False, **kwargs)
        return test_data, test_loader
    else:
        _, train_loader = _prepare_dataset(
                filepath=config["train_file"], shuffle=True, **kwargs)
        _, val_loader = _prepare_dataset(
                filepath=config["val_file"], shuffle=False, **kwargs)
        return train_loader, val_loader
