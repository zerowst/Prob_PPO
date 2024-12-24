from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams
from difflib import SequenceMatcher
import numpy as np
import time
from pathlib import Path
from typing import Any, Dict, Tuple, Callable, List, Optional, Union



############################################################################################################################################
alpha = 0.7


def softmax(data, lens):
    data = np.array(data)
    lens = np.array(lens)
    data = data/(2*lens**alpha)         # 平滑,减小差异
    exp_x = np.exp(data - np.max(data, axis=-1, keepdims=True))  # 防止溢出
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def get_weight_sum_value(rollouts: List, rewards: List):
    lens = [len for _, _, _, len in rollouts]
    probs = [prob for _, _, prob, _ in rollouts]
    probs = softmax(probs,lens)
    print(probs)
    value = np.dot(probs,np.array(rewards))
    return value
############################################################################################################################################


### -----------------------------------------------Computing similarity for samples-----------------------------------------  


def sample_similarity(answers: List[str]):        
        # Pairwise n-gram Jaccard Similarity
    def ngram_jaccard_similarity(answer1, answer2, n=3):
        def pairwise_jaccard(answer1, answer2, n=3):
            set1 = set(ngrams(answer1.split(), n))
            set2 = set(ngrams(answer2.split(), n))
            return len(set1 & set2) / len(set1 | set2)

        n_gram_similarities = [
            pairwise_jaccard(answers[i], answers[j])
            for i in range(len(answers)) for j in range(i + 1, len(answers))
        ]
        mean_ngram_similarity = np.mean(n_gram_similarities)
        return mean_ngram_similarity
    
    # TFIDF similarity
    def tfidf_similarity(answers):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(answers)
        cosine_similarities = cosine_similarity(tfidf_matrix)
        if cosine_similarities.shape[0] == 1:
            return 1
        mean_tfidf_similarity = np.mean(cosine_similarities[np.triu_indices_from(cosine_similarities, k=1)])
        return mean_tfidf_similarity
    
    similarity = tfidf_similarity(answers)

    return similarity


def get_similarity(rollouts: List, rewards: List):

    # rollouts contains answers for each step Reward was computed, get the correct step idx
    LENGTH = len(rewards)
    correct_step_idx = [i for i in range(LENGTH) if rewards[i] == 1]
    #
    # rollouts contains children of tree, 9 samples
    # get the correct samples whoes reward are 1    

    len_correct = len(correct_step_idx)
    if len_correct != 0 and len_correct != LENGTH:
        correct_cases = [rollouts[i][0] for i in correct_step_idx]
        wrong_cases = [rollouts[i][0] for i in range(LENGTH) if rewards[i] == 0]

        # get similarity within positive cases

        positive_score = sample_similarity(correct_cases)


        # get similarity of negative cases to positive cases
        similarity = []
        for wrong in wrong_cases:
            local_sim = []
            
            for correct in correct_cases:
                local_sim.append(sample_similarity([correct, wrong]))
            
            similarity.append(max(local_sim))

        negative_score = sum(similarity) / len(similarity)

    # all wrong cases, no correct case -> positive score=0, negative=0
    elif len_correct == 0:
        positive_score = 0
        negative_score = 0
    # all correct cases, positive=1, negative=1
    else:
        positive_score = 1
        negative_score = 1

    mc = sum(rewards) / len(rewards)*(1 + positive_score * 0.2) - \
         (len(rewards) - sum(rewards))/ len(rewards) * 0.2/2 * (1-negative_score)
            
    print(mc)
    return [positive_score, negative_score]


def test_sim():
    rollouts = [('follow the previous steps', 0), ('following the steps', 9), ('follow it', 9), ('next', 1)]
    rewards = [1.0, 0.0, 1.0, 0.0]
    s = get_similarity(rollouts, rewards)
    print(s)


def test_weight():
    rollouts = [('s', 's', 0.36, 15.0), ('s', 's', -3.36, 55.0), ('s', 's', -0.36, 25.0), ('s', 's', 0.0, 9.0)]
    rewards = [0.0, 1.0, 0.0, 1.0]
    v = get_weight_sum_value(rollouts, rewards)
    print(v)
   



if __name__ == '__main__':
    
    # test_sim()

    test_sim()




