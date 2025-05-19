
"""
Evaluation module for GraphRAG.
This module handles evaluation metrics for the question answering system.
"""

from typing import List, Dict, Any
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score

class Evaluator:
    def __init__(self):
        """Initialize the evaluator."""
        try:
            import nltk
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK punkt tokenizer: {str(e)}")
    
    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """
        Calculate BLEU score between reference and generated answer.
        
        Args:
            reference (str): The reference (ground truth) answer
            hypothesis (str): The generated answer to evaluate
            
        Returns:
            float: BLEU score
        """
        if not reference or not hypothesis:
            return 0.0
        
        # Tokenize the sentences
        reference_tokens = word_tokenize(reference.lower())
        hypothesis_tokens = word_tokenize(hypothesis.lower())
        
        # BLEU requires a list of references
        references = [reference_tokens]
        
        # Use smoothing to avoid zero scores when there are no matches
        smoothing = SmoothingFunction().method1
        
        # Calculate BLEU score (with weights for unigrams, bigrams, trigrams, and 4-grams)
        try:
            # Calculate BLEU-1 (unigrams only), which is more lenient
            bleu1 = sentence_bleu(references, hypothesis_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            
            # Calculate BLEU-4 (standard with all n-grams)
            bleu4 = sentence_bleu(references, hypothesis_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            
            return {'bleu1': bleu1, 'bleu4': bleu4}
        except Exception as e:
            print(f"Error calculating BLEU score: {str(e)}")
            return {'bleu1': 0.0, 'bleu4': 0.0}
    
    def calculate_exact_match(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Exact Match (EM) score.
        
        Args:
            reference (str): The reference (ground truth) answer
            hypothesis (str): The generated answer to evaluate
            
        Returns:
            float: 1.0 if answers match exactly (after normalization), 0.0 otherwise
        """
        # Normalize answers by converting to lowercase and removing extra spaces
        def normalize(text):
            return ' '.join(text.lower().split())
        
        return float(normalize(reference) == normalize(hypothesis))
    
    def calculate_f1(self, reference: str, hypothesis: str) -> float:
        """
        Calculate token-level F1 score between reference and generated answer.
        
        Args:
            reference (str): The reference (ground truth) answer
            hypothesis (str): The generated answer to evaluate
            
        Returns:
            float: F1 score
        """
        if not reference or not hypothesis:
            return 0.0
        
        # Tokenize and convert to sets
        ref_tokens = set(word_tokenize(reference.lower()))
        hyp_tokens = set(word_tokenize(hypothesis.lower()))
        
        # Calculate precision, recall, and F1
        if not hyp_tokens:
            return 0.0
        if not ref_tokens:
            return 0.0
        
        true_positives = len(ref_tokens & hyp_tokens)
        precision = true_positives / len(hyp_tokens) if hyp_tokens else 0.0
        recall = true_positives / len(ref_tokens) if ref_tokens else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1
    
    def evaluate(self, references: List[str], hypotheses: List[str]) -> Dict[str, Any]:
        """
        Evaluate a list of answer pairs using multiple metrics.
        
        Args:
            references (List[str]): List of reference (ground truth) answers
            hypotheses (List[str]): List of generated answers to evaluate
            
        Returns:
            Dict: Dictionary of evaluation results
        """
        if len(references) != len(hypotheses):
            raise ValueError("References and hypotheses lists must have the same length")
        
        # Initialize metrics
        bleu1_scores = []
        bleu4_scores = []
        f1_scores = []
        em_scores = []
        
        # Calculate metrics for each pair
        for ref, hyp in zip(references, hypotheses):
            bleu = self.calculate_bleu(ref, hyp)
            bleu1_scores.append(bleu['bleu1'])
            bleu4_scores.append(bleu['bleu4'])
            f1_scores.append(self.calculate_f1(ref, hyp))
            em_scores.append(self.calculate_exact_match(ref, hyp))
        
        # Aggregate results
        results = {
            'bleu1': np.mean(bleu1_scores),
            'bleu4': np.mean(bleu4_scores),
            'f1': np.mean(f1_scores),
            'exact_match': np.mean(em_scores),
            'sample_count': len(references),
            'individual_scores': [
                {
                    'reference': ref,
                    'hypothesis': hyp,
                    'bleu1': b1,
                    'bleu4': b4,
                    'f1': f1,
                    'exact_match': em
                }
                for ref, hyp, b1, b4, f1, em in zip(references, hypotheses, bleu1_scores, bleu4_scores, f1_scores, em_scores)
            ]
        }
        
        return results
