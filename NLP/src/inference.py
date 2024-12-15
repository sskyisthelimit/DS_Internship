# i was really confused by definition of inference file
# please better check out demo notebook which also is a test report notebook

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

labels = ["O", "B-MOUNTAIN", "I-MOUNTAIN", "B-ELEVATION", "I-ELEVATION"]
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("sskyisthelimit/mount-ner-model")
model = AutoModelForTokenClassification.from_pretrained("sskyisthelimit/mount-ner-model")


def post_process_tokens_and_labels(tokens, predicted_labels):
    """
    Post-process tokens and labels by merging subwords and removing special tokens like [CLS] and [SEP].

    Args:
    tokens (list): List of tokenized input words.
    predicted_labels (list): List of predicted labels.

    Returns:
    final_tokens (list): List of final tokens with subwords merged.
    final_labels (list): List of final labels corresponding to each token.
    """
    final_tokens, final_labels = [], []

    for token, label in zip(tokens, predicted_labels):
        if token not in ['[CLS]', '[SEP]']:  # Skip special tokens
            if token.startswith("##"):
                final_tokens[-1] += token[2:]  # Merge subwords
            else:
                final_tokens.append(token)
                final_labels.append(label)

    return final_tokens, final_labels


def predict_ner_labels(sentence, model, tokenizer, device, print_pairs=False):
    """
    Predicts NER labels for a given sentence using the fine-tuned model and tokenizer.

    Args:
    sentence (str): The input sentence.
    model
    tokenizer
    device
    label_map (dict): Mapping from label IDs to label names.
    """

    # tokenize
    inputs = tokenizer(sentence.split(),
                       return_tensors="pt", is_split_into_words=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs).logits

    # predicted labels
    predictions = outputs.argmax(dim=-1).squeeze().tolist()
    predicted_labels = [pred for pred in predictions]

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"].squeeze().tolist())
    # post-process
    final_tokens, final_labels = post_process_tokens_and_labels(
        tokens, predicted_labels)
    
    if print_pairs:
        for token, label in final_labels:
            print(f"{token}: {id2label[label]}")

    return final_tokens, final_labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Example of parsing multiple named arguments.')

    parser.add_argument('--sentence', type=str,
                        help='A sentence string', required=True)
    parser.add_argument('--device', type=str,
                        help='A device string', required=True)

    args = parser.parse_args()

    predict_ner_labels(
        args.sentence,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        print_pairs=True)
