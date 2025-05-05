from typing import List, Tuple, Set
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim.downloader import load
import nltk
import string
import numpy as np
import logging

try:
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading stopwords: {e}")

logger = logging.getLogger(__name__)


class RoMAPerturbator:
    def __init__(self, epsilon: float = 0.35):
        """
        Initialize perturbator with stricter epsilon control.

        Args:
            epsilon: Controls the size of the perturbation ball (0.0 to 1.0)
                    Lower values = more similar words
                    Higher values = more diverse words
        """
        self.epsilon = epsilon
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r"[A-Za-z]+[']*[A-Za-z]*|\S+")

        try:
            print("Loading word2vec-google-news-300 embeddings...")
            self.word_vectors = load('word2vec-google-news-300')
            self.has_embeddings = True
            print("Successfully loaded word2vec embeddings.")
        except Exception as e:
            print(f"Could not load word embeddings: {e}")
            self.word_vectors = None
            self.has_embeddings = False

    def get_embedding_replacements(self, word: str, topn: int = None) -> List[str]:
        if not self.has_embeddings:
            logger.debug(f"[get_embedding_replacements] No word embeddings. Returning empty list.")
            return []

        try:
            base_word = word[:-2] if word.endswith("'s") else word

            # Increase topn dynamically
            topn = max(250, int(400 * self.epsilon))

            similar_words = self.word_vectors.most_similar(base_word.lower(), topn=topn)

            #logger.debug(
            #    f"[get_embedding_replacements] Similar words for '{word}' (base '{base_word}'): {similar_words}")

            threshold = 1.0 - self.epsilon
            valid_replacements = []
            total_candidates = 0
            passed_candidates = 0

            for similar_word, score in similar_words:
                total_candidates += 1
                if score < threshold:
                    # Log rejections below threshold at debug level
                    logger.debug(
                        f"[get_embedding_replacements] Rejected '{similar_word}' with score {score:.3f} < {threshold:.3f}")
                    continue

                if word.endswith("'s"):
                    similar_word += "'s"

                # Preserve capitalization
                if word.istitle():
                    similar_word = similar_word.title()
                elif word.isupper():
                    similar_word = similar_word.upper()

                if self.is_valid_replacement(similar_word, word):
                    passed_candidates += 1
                    valid_replacements.append((similar_word, score))
                else:
                    # Log invalid replacements for debugging
                    logger.debug(f"[get_embedding_replacements] Invalid replacement '{similar_word}' for '{word}'")

            # Sort valid_replacements by their (score) in descending order
            valid_replacements.sort(key=lambda x: x[1], reverse=True)

            # Keep 100%
            num_to_keep = int(len(valid_replacements) * 1.0)
            trimmed = [r[0] for r in valid_replacements[:num_to_keep]]

            #logger.info(
            #    f"[get_embedding_replacements] Word '{word}' → found {total_candidates} candidates, "
            #    f"{passed_candidates} passed threshold & validation, returning {len(trimmed)}."
            #)

            return trimmed

        except KeyError:
            logger.warning(f"[get_embedding_replacements] KeyError for word '{word}'. Returning empty list.")
            return []

    def generate_perturbations(self, sentence: str, max_perturbations: int = 1000) -> List[Tuple[List[int], str]]:
        tokens = self.tokenizer.tokenize(sentence)
        perturbations = []
        seen_perturbations = set()

        valid_positions = [
            i for i, token in enumerate(tokens)
            if any(c.isalpha() for c in token) and token.lower() not in {"a", "of", "and", "'s"}
        ]

        if not valid_positions:
            logger.debug(f"[generate_perturbations] No valid positions in sentence: {sentence}")
            return perturbations

        num_valid_positions = len(valid_positions)
        replacements_per_word = max(10, min(100, max_perturbations // num_valid_positions))

        # Randomly select some subset of positions to spread out perturbations
        chosen_positions = np.random.choice(valid_positions, size=min(num_valid_positions, 10), replace=False)
        logger.info(
            f"[generate_perturbations] Processing sentence: '{sentence[:40]}...', "
            f"valid_positions={num_valid_positions}, chosen_positions={list(chosen_positions)}, "
            f"replacements_per_word={replacements_per_word}"
        )

        for pos in chosen_positions:
            original_word = tokens[pos]
            replacements = self.get_embedding_replacements(original_word)

            if not replacements:
                logger.debug(f"[generate_perturbations] No replacements for '{original_word}'")
                continue

            # Randomly sample from the pool of valid replacements
            to_use = min(len(replacements), replacements_per_word)
            selected_replacements = np.random.choice(replacements, size=to_use, replace=False)

            for replacement in selected_replacements:
                new_tokens = tokens.copy()
                new_tokens[pos] = replacement
                perturbed_sentence = self.reconstruct_sentence(new_tokens)

                if perturbed_sentence != sentence and perturbed_sentence not in seen_perturbations:
                    perturbations.append(([pos], perturbed_sentence))
                    seen_perturbations.add(perturbed_sentence)

                    if len(perturbations) >= max_perturbations:
                        logger.debug("[generate_perturbations] Reached max_perturbations limit.")
                        return perturbations

        logger.info(
            f"[generate_perturbations] Final count of perturbations for sentence: {len(perturbations)}"
        )

        return perturbations

    def is_valid_replacement(self, replacement: str, original: str) -> bool:
        """
        Validate if a replacement word is acceptable.

        Args:
            replacement: The proposed replacement word
            original: The original word being replaced

        Returns:
            Boolean indicating if replacement is valid
        """
        if not replacement or len(replacement) < 2:
            return False

        base_replacement = replacement[:-2] if replacement.endswith("'s") else replacement
        base_original = original[:-2] if original.endswith("'s") else original

        if base_replacement.lower() == base_original.lower():
            return False

        # Always be strict with stopwords
        if base_replacement.lower() in self.stop_words and base_original.lower() not in self.stop_words:
            return False

        # Stricter format validation
        valid_format = (
                base_replacement.isalpha() or
                (base_replacement.replace("'", "").isalpha() and "'" in base_replacement)
        )

        return valid_format

    def reconstruct_sentence(self, tokens: List[str]) -> str:
        """
        Reconstruct a sentence from tokens while preserving formatting.

        Args:
            tokens: List of tokens to combine

        Returns:
            Reconstructed sentence string
        """
        result = ""
        for i, token in enumerate(tokens):
            if i == 0:
                result = token
            else:
                prev_token = tokens[i - 1]

                if (token in string.punctuation or
                        token == "'" or
                        token.startswith("-") or
                        prev_token.endswith("-")):
                    result += token
                else:
                    result += " " + token

        return result.strip()