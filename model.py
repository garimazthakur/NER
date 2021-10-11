from transformers import AutoModelForTokenClassification


def model_init(model_name, label_map):
    return AutoModelForTokenClassification.from_pretrained(model_name, return_dict=True, num_labels=len(label_map))

