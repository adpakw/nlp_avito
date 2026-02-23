import numpy as np
from seqeval.metrics import f1_score

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


# def align_labels_with_tokens(
#     labels: list[int], tokens: list[str], special_tokens: list[str]
# ):
#     aligned_labels = []
#     word_index = 0

#     for token in tokens:
#         if token in special_tokens:
#             aligned_labels.append(-100)
#             continue

#         current_label = labels[word_index]

#         print(word_index)


#         if token.startswith("##"):
#             if current_label in [1, 3, 5, 7]:  # начала сущностей
#                 aligned_labels.append(current_label + 1)  # продолжения сущностей
#             elif current_label in [2, 4, 6, 8]:  # продолжения сущностей
#                 aligned_labels.append(current_label)
#             else:
#                 aligned_labels.append(0)
#         else:
#             aligned_labels.append(current_label)
#             word_index += 1

#     return aligned_labels


def align_labels_with_tokens(labels: list[int], word_ids: list[int]):
    result = []

    prev_word_id = None

    for word_id in word_ids:

        if word_id is None:
            result.append(-100)

        elif word_id != prev_word_id:
            result.append(labels[word_id])

        else:
            cur_label_id = labels[word_id] + 1 if labels[word_id] in [1, 3, 5, 7] else 0
            result.append(cur_label_id)

        prev_word_id = word_id

    return result


def f1_ner(eval_pred):
    pred_proba, labels = eval_pred
    preds = np.argmax(pred_proba, axis=-1)

    str_preds = []
    str_labels = []

    for pred, label in zip(preds, labels):
        mask = label != -100

        str_preds.append([LABELS[p] for p in pred[mask]])
        str_labels.append([LABELS[l] for l in label[mask]])

    return {"f1": f1_score(str_preds, str_labels)}
