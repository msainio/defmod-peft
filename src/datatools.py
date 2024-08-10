import logging
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

logger = logging.getLogger(__name__)

def _get_input_texts(colname_def, colname_ex, colname_word, data):
    str_data = data.map(lambda x: str(x))
    input_texts = (
            str_data[colname_ex] + " What is the definition of " +
            str_data[colname_word] + "? " + str_data[colname_def]
            ).to_list()
    return input_texts

def _get_labels(model_inputs, pad_token_id):
    labels = model_inputs["input_ids"].clone().detach()
    labels[labels == pad_token_id] = -100
    return labels

def _get_model_inputs(colname_def, colname_ex, colname_word, data, max_length,
        tokenizer):
    input_texts = _get_input_texts(
            colname_def, colname_ex, colname_word, data)
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
        batch_size, colname_def, colname_ex, colname_word, filepath,
        max_length, num_workers, pin_memory, shuffle, tokenizer):
    data = pd.read_csv(filepath)
    model_inputs = _get_model_inputs(
            colname_def, colname_ex, colname_word,
            data, max_length, tokenizer)
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
    return dataloader

def prepare_datasets(config, num_workers, pin_memory, tokenizer):
    shared_kwargs = {
            "batch_size": config["batch_size"],
            "max_length": config["max_length"],
            "colname_def": config["colname_def"],
            "colname_ex": config["colname_ex"],
            "colname_word": config["colname_word"],
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "tokenizer": tokenizer,
            }
    train_set = _prepare_dataset(
            filepath=config["train_file"], shuffle=True, **shared_kwargs)
    val_set = _prepare_dataset(
            filepath=config["val_file"], shuffle=False, **shared_kwargs)
    return train_set, val_set
