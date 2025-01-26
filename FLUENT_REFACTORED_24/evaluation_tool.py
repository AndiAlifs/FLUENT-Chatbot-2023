from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import math
from typing import List

def calculate_bleu(preds, questions, answers):
    bleu_score_1 = 0
    bleu_score_2 = 0
    bleu_score_3 = 0
    bleu_score_4 = 0
    cum_bleu_score_1 = 0
    cum_bleu_score_2 = 0
    cum_bleu_score_3 = 0
    cum_bleu_score_4 = 0

    num_of_rows_calculated = 0
    smoothing_function = SmoothingFunction().method1

    for i, (question, real_answer) in enumerate(zip(questions, answers)):
        # print(f"Question: {question}")
        # print(f"Real Answer: {real_answer}")
        # print(f"Predicted Answer: {preds[i]}")
        refs = [real_answer.split(' ')]
        hyp = preds[i].split(' ')

        bleu_score_1 += sentence_bleu(refs, hyp, weights=(1,0,0,0), smoothing_function=smoothing_function)
        bleu_score_2 += sentence_bleu(refs, hyp, weights=(0,1,0,0), smoothing_function=smoothing_function)
        bleu_score_3 += sentence_bleu(refs, hyp, weights=(0,0,1,0), smoothing_function=smoothing_function)
        bleu_score_4 += sentence_bleu(refs, hyp, weights=(0,0,0,1), smoothing_function=smoothing_function)
        cum_bleu_score_1 += sentence_bleu(refs, hyp, weights=(1,0,0,0), smoothing_function=smoothing_function)
        cum_bleu_score_2 += sentence_bleu(refs, hyp, weights=(0.5,0.5,0,0), smoothing_function=smoothing_function)
        cum_bleu_score_3 += sentence_bleu(refs, hyp, weights=(0.33,0.33,0.33,0), smoothing_function=smoothing_function)
        cum_bleu_score_4 += sentence_bleu(refs, hyp, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothing_function)

        num_of_rows_calculated+=1

    results = {"1-gram": (bleu_score_1/num_of_rows_calculated),
                "2-gram": (bleu_score_2/num_of_rows_calculated),
                "3-gram": (bleu_score_3/num_of_rows_calculated),
                "4-gram": (bleu_score_4/num_of_rows_calculated),
                "cumulative-1-gram": (cum_bleu_score_1/num_of_rows_calculated),
                "cumulative-2-gram": (cum_bleu_score_2/num_of_rows_calculated),
                "cumulative-3-gram": (cum_bleu_score_3/num_of_rows_calculated),
                "cumulative-4-gram": (cum_bleu_score_4/num_of_rows_calculated)}

    return results

def count_bleu_score(model_count, real_answers, questions):
    preds = []
    for question in questions:
        outputs = model_count.generate(question)
        decoded_output = model_count.dec_tokenizer.decode(outputs[0])
        preds.append(decoded_output)
    return calculate_bleu(preds, questions, real_answers)

def preprocess_text(text):
    # Normalize the text by converting to lowercase and stripping whitespace
    return text.lower().strip()

def get_character_ngrams(text, n):
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    return ngrams

def count_ngrams(text, max_order):
    ngram_counts = {}
    for n in range(1, max_order+1):
        ngrams = get_character_ngrams(text, n)
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    return ngram_counts

def compute_precision_recall(candidate_counts, reference_counts):
    overlap = 0
    total_candidate = sum(candidate_counts.values())
    total_reference = sum(reference_counts.values())

    for ngram in candidate_counts:
        if ngram in reference_counts:
            overlap += min(candidate_counts[ngram], reference_counts[ngram])

    precision = overlap / total_candidate if total_candidate > 0 else 0.0
    recall = overlap / total_reference if total_reference > 0 else 0.0

    return precision, recall

def compute_chrf(candidate: str, references: List[str], max_order=6, beta=2):
    candidate = preprocess_text(candidate)
    candidate_ngram_counts = count_ngrams(candidate, max_order)

    # Aggregate n-gram counts from all references
    reference_ngram_counts = {}
    reference = preprocess_text(references)
    reference_counts = count_ngrams(reference, max_order)
    for ngram, count in reference_counts.items():
        if ngram in reference_ngram_counts:
            reference_ngram_counts[ngram] = max(reference_ngram_counts[ngram], count)
        else:
            reference_ngram_counts[ngram] = count

    precision, recall = compute_precision_recall(candidate_ngram_counts, reference_ngram_counts)

    beta_squared = beta ** 2
    if precision + recall > 0:
        chrf_score = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
    else:
        chrf_score = 0.0

    # Multiply by 100 to get a percentage
    return chrf_score * 100

def compute_chrf_list(list_of_candidates: List[str], list_of_references: List[str], max_order=6, beta=2):
    if len(list_of_candidates) != len(list_of_references):
        raise ValueError("The number of candidates and reference lists must be the same.")

    scores = []
    for candidate, references in zip(list_of_candidates, list_of_references):
        score = compute_chrf(candidate, references, max_order, beta)
        scores.append(score)
    return scores

def compute_average_chrf(list_of_candidates: List[str], list_of_references: List[str], max_order=6, beta=2):
    scores = compute_chrf_list(list_of_candidates, list_of_references, max_order, beta)
    return sum(scores) / len(scores)

def generate_predictions(model_count, questions):
    preds = []
    for question in questions:
        outputs = model_count.generate(question)
        decoded_output = model_count.dec_tokenizer.decode(outputs[0])
        preds.append(decoded_output)
    return preds