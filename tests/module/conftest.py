"""conftest file."""

supported_model_names = [
    "bert-base-uncased",
    "google/vit-base-patch16-224",
    "gpt2",
    "microsoft/resnet-152",
    "t5-small",
]

# TODO: remove this; make the input names determined from a sample dataset
#       by using oobleck's dataset.py
input_names = [
    ["input_ids", "token_type_ids", "attention_mask", "labels"],
    ["pixel_values", "labels"],
    ["input_ids", "attention_mask", "labels"],
    ["pixel_values", "labels"],
    ["input_ids", "attention_mask", "labels"],
]

model_input_names_pairs = list(zip(supported_model_names, input_names))
