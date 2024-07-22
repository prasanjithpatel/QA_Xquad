import re
from collections import Counter 


# Normalization functions
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))

    return white_space_fix(remove_articles(remove_punc(s)))

# Token F1 Score function
def token_f1_score(pred, gold):
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1*100

# Exact Match Score function
def exact_match_score(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))*100


