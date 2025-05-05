import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.tag import pos_tag
from datasets import load_dataset
import string
import itertools
from typing import List, Tuple, Set, Dict
import random
from gensim.models import KeyedVectors
from collections import Counter
import math
import os
from datasets import load_dataset


def download_nltk_resources():
    """Download all required NLTK resources"""
    resources = ['punkt', 'averaged_perceptron_tagger','averaged_perceptron_tagger_eng', 'wordnet', 'stopwords', 'brown']
    for resource in resources:
        try:
            nltk.download(resource)
            print(f"Successfully downloaded {resource}")
        except Exception as e:
            print(f"Error downloading {resource}: {str(e)}")
            raise


# Download required NLTK resources
download_nltk_resources()


class RelevancyFilter:
    def __init__(self):
        """Initialize with fallback if Brown corpus isn't available"""
        try:
            from nltk.corpus import brown
            from collections import Counter
            self.word_frequencies = Counter(brown.words())
            self.total_words = sum(self.word_frequencies.values())
            self.has_frequencies = True
        except LookupError:
            print("Warning: Brown corpus not available. Using simplified scoring.")
            self.has_frequencies = False

    def get_word_commonness(self, word: str) -> float:
        """Calculate how common a word is, with fallback"""
        if not self.has_frequencies:
            # Fallback scoring based on word characteristics
            return 1.0 - (0.1 * (len(word) > 7))  # Slight penalty for longer words

        freq = self.word_frequencies[word.lower()]
        return math.log(freq + 1) / math.log(self.total_words) if freq > 0 else 0

    def score_synonym(self, synonym: str, original_word: str) -> float:
        """Score a synonym's relevance with fallback scoring"""
        score = 0.0

        # Word commonness (if available)
        if self.has_frequencies:
            score += self.get_word_commonness(synonym) * 2.0

        # Length similarity (always available)
        length_diff = abs(len(synonym) - len(original_word))
        score += 1.0 / (length_diff + 1)

        # Penalize technical terms (always available)
        if any(synonym.endswith(suffix) for suffix in
               ['oid', 'ation', 'ology', 'osis', 'itis']):
            score *= 0.5

        return score

class ModernSynonymGenerator:
    def __init__(self):
        # Load pre-trained word vectors (you would need to download these)
        # self.word_vectors = KeyedVectors.load_word2vec_format('word2vec.bin', binary=True)
        self.word_vectors = None  # Initialize in practice

    def get_synonyms(self, word: str) -> Set[str]:
        """Get synonyms using multiple modern sources"""
        synonyms = set()

        # 1. Get WordNet synonyms
        wordnet_syns = self._get_wordnet_synonyms(word)
        synonyms.update(wordnet_syns)

        # 2. Get word vector synonyms (if vectors are loaded)
        if self.word_vectors and word in self.word_vectors:
            vector_syns = self._get_vector_synonyms(word)
            synonyms.update(vector_syns)

        # 3. Get common usage pairs (this could be loaded from a curated dataset)
        # common_syns = self._get_common_usage_synonyms(word)
        # synonyms.update(common_syns)

        return synonyms

    def _get_wordnet_synonyms(self, word: str) -> Set[str]:
        """Get traditional WordNet synonyms"""
        synonyms = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.add(lemma.name())
        return synonyms

    def _get_vector_synonyms(self, word: str, topn: int = 10,
                             threshold: float = 0.7) -> Set[str]:
        """Get synonyms based on word vector similarity"""
        if not self.word_vectors or word not in self.word_vectors:
            return set()

        similar_words = self.word_vectors.most_similar(word, topn=topn)
        return {w for w, score in similar_words if score > threshold}


class ContextAwarePerturbator:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # self.tokenizer = RegexpTokenizer(r'\w+')
        self.tokenizer = RegexpTokenizer(r"[A-Za-z]+[']*[A-Za-z]*|\S+")

        # POS mapping between NLTK and WordNet tags
        self.pos_map = {
            'NN': wn.NOUN, 'NNS': wn.NOUN, 'NNP': wn.NOUN, 'NNPS': wn.NOUN,
            'VB': wn.VERB, 'VBD': wn.VERB, 'VBG': wn.VERB, 'VBN': wn.VERB, 'VBP': wn.VERB, 'VBZ': wn.VERB,
            'JJ': wn.ADJ, 'JJR': wn.ADJ, 'JJS': wn.ADJ,
            'RB': wn.ADV, 'RBR': wn.ADV, 'RBS': wn.ADV
        }

    def get_base_word(self, word: str) -> str:
        """Extract base word from possible possessive form"""
        if word.endswith("'s"):
            return word[:-2]
        return word

    def get_word_context(self, sentence: str) -> Dict[str, str]:
        """Get POS tags using NLTK's pos_tag function"""
        # Use the tokenizer for consistent tokenization
        tokens = self.tokenizer.tokenize(sentence)

        # Use NLTK's pos_tag for more accurate tagging
        tagged = pos_tag(tokens)

        # Create context dictionary
        context_dict = {word: tag for word, tag in tagged}

        # For words not tagged, use fallback system
        for token in tokens:
            if token not in context_dict:
                if token.endswith('ly'):
                    context_dict[token] = 'RB'
                elif token.endswith(('ed', 'ing')):
                    context_dict[token] = 'VB'
                elif token.endswith(('ment', 'tion', 'ness', 'ity')):
                    context_dict[token] = 'NN'
                elif token.endswith(('ful', 'ous', 'ible', 'able', 'al')):
                    context_dict[token] = 'JJ'
                else:
                    context_dict[token] = 'NN'

        return context_dict

    def get_broad_synonyms(self, word: str, pos_tag: str) -> Set[str]:
        """Get broader set of synonyms by exploring WordNet relationships"""
        synonyms = set()
        wordnet_pos = self.pos_map.get(pos_tag)

        # Get all synsets for the word
        synsets = wn.synsets(word)
        if wordnet_pos:
            synsets = [syn for syn in synsets if syn.pos() == wordnet_pos]

        for synset in synsets:
            # Add direct synonyms
            synonyms.update(lemma.name() for lemma in synset.lemmas() if lemma.name() != word)

            # Add hypernyms (more general terms)
            for hypernym in synset.hypernyms():
                synonyms.update(lemma.name() for lemma in hypernym.lemmas())

            # Add hyponyms (more specific terms)
            for hyponym in synset.hyponyms():
                synonyms.update(lemma.name() for lemma in hyponym.lemmas())

            # Add sister terms (share same hypernym)
            for hypernym in synset.hypernyms():
                for hyponym in hypernym.hyponyms():
                    synonyms.update(lemma.name() for lemma in hyponym.lemmas())

            # Add similar_tos (specifically for adjectives)
            for similar in synset.similar_tos():
                synonyms.update(lemma.name() for lemma in similar.lemmas())

            # Add also_sees (related terms)
            for also in synset.also_sees():
                synonyms.update(lemma.name() for lemma in also.lemmas())

        # Clean up synonyms
        synonyms = {s for s in synonyms if '_' not in s and '-' not in s}  # Remove multi-word expressions
        synonyms = {s for s in synonyms if len(s) > 1}  # Remove single letters

        return synonyms

    def is_basic_context_valid(self, sentence: str) -> bool:
        """Perform very basic context validation"""
        # Just check for basic grammatical markers
        basic_markers = {
            ('a', 'an'): {'a', 'an', 'the'},  # Articles
            ('is', 'are'): {'is', 'are', 'was', 'were'},  # Be verbs
            ('this', 'that'): {'this', 'that', 'these', 'those'}  # Demonstratives
        }

        words = self.tokenizer.tokenize(sentence.lower())

        # Check if article usage is reasonable
        for i, word in enumerate(words[:-1]):
            if word in {'a', 'an'}:
                # Very basic check for vowel sound
                next_word = words[i + 1]
                if word == 'a' and next_word[0] in 'aeiou':
                    return False
                if word == 'an' and next_word[0] not in 'aeiou':
                    return False

        return True

    def get_context_aware_synonyms(self, word: str, pos_tag: str, sentence: str) -> Set[str]:
        """Get synonyms that match the word's context and POS"""
        # Initialize relevancy filter if not exists
        if not hasattr(self, 'relevancy_filter'):
            self.relevancy_filter = RelevancyFilter()

        synonyms = set()

        # Get WordNet POS tag
        wordnet_pos = self.pos_map.get(pos_tag, wn.NOUN)  # Default to NOUN if no mapping

        # Get all synsets
        synsets = wn.synsets(word, pos=wordnet_pos)

        # If no synsets found with POS filtering, try without POS filter
        if not synsets:
            synsets = wn.synsets(word)

        # Collect synonyms from synsets
        for synset in synsets[:3]:  # Consider top 3 meanings
            for lemma in synset.lemmas():
                synonym = lemma.name()
                if synonym != word and '_' not in synonym:
                    synonyms.add(synonym)

            # Add selected similar words
            for similar in synset.similar_tos()[:2]:
                for lemma in similar.lemmas():
                    synonym = lemma.name()
                    if synonym != word and '_' not in synonym:
                        synonyms.add(synonym)

        # Score and filter synonyms by relevance
        if synonyms:
            synonym_scores = [
                (syn, self.relevancy_filter.score_synonym(syn, word))
                for syn in synonyms
            ]
            synonym_scores.sort(key=lambda x: x[1], reverse=True)
            synonyms = {s[0] for s in synonym_scores[:10]}  # Keep top 10 most relevant

        # Apply context filtering
        filtered_synonyms = set()
        original_context = self.get_sentence_context(sentence, word)

        for synonym in synonyms:
            test_sentence = self.create_test_sentence(sentence, word, synonym)
            synonym_context = self.get_sentence_context(test_sentence, synonym)

            if self.is_context_similar(original_context, synonym_context):
                filtered_synonyms.add(synonym)

        return filtered_synonyms

    def get_sentence_context(self, sentence: str, target_word: str) -> Set[str]:
        """Get focused context around target word"""
        tokens = self.tokenizer.tokenize(sentence.lower())
        try:
            word_index = tokens.index(target_word.lower())

            # Reduce window size to immediate neighbors
            start = max(0, word_index - 2)  # Reduced from 3 to 2
            end = min(len(tokens), word_index + 3)  # Reduced from 4 to 3

            # Get immediate context only
            context_words = set(tokens[start:end]) - {target_word.lower()}

            # Focus on structural words that indicate grammatical role
            structural_indicators = {'a', 'an', 'the', 'this', 'that', 'these', 'those',
                                     'my', 'your', 'his', 'her', 'its', 'our', 'their'}

            # Keep structural words and immediate neighbors
            important_context = {word for word in context_words
                                 if word in structural_indicators or
                                 abs(tokens.index(word) - word_index) <= 1}

            return important_context
        except ValueError:
            return set()

    def create_test_sentence(self, original: str, old_word: str, new_word: str) -> str:
        """Create a test sentence by replacing old_word with new_word using robust tokenization"""
        try:
            # Primary tokenization attempt using word_tokenize
            tokens = word_tokenize(original)
        except LookupError:
            # Fallback to RegexpTokenizer if word_tokenize fails
            tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
            tokens = tokenizer.tokenize(original)

        return ' '.join(new_word if t.lower() == old_word.lower() else t for t in tokens)

    def is_context_similar(self, context1: Set[str], context2: Set[str]) -> bool:
        """Simplified context similarity check focusing on structural similarity"""

        # Focus only on structural words and immediate neighbors
        def get_key_words(context):
            # Keep determiners, possessives, and content words
            structural = {'a', 'an', 'the', 'this', 'that', 'these', 'those',
                          'my', 'your', 'his', 'her', 'its', 'our', 'their'}
            return {w for w in context if w in structural or len(w) > 3}

        key_words1 = get_key_words(context1)
        key_words2 = get_key_words(context2)

        # If either context has structural words, require at least one match
        if key_words1 and key_words2:
            return bool(key_words1.intersection(key_words2))

        # If no structural words, contexts are considered similar
        return True

    def get_enhanced_synonyms(self, word: str, sentence: str) -> List[str]:
        """Get enhanced list of synonyms using context awareness"""
        # Get POS context
        word_contexts = self.get_word_context(sentence)
        pos_tag = word_contexts.get(word, '')

        # Get context-aware synonyms
        synonyms = self.get_context_aware_synonyms(word, pos_tag, sentence)

        # Add common word variations while preserving case
        if word.istitle():
            synonyms.update(s.title() for s in synonyms)
        if word.isupper():
            synonyms.update(s.upper() for s in synonyms)

        # Handle word variations based on POS
        if pos_tag.startswith('VB'):
            # Handle verb forms
            base_synonyms = set(synonyms)
            if word.endswith('ing'):
                synonyms.update(s + 'ing' for s in base_synonyms)
            elif word.endswith('ed'):
                synonyms.update(s + 'ed' for s in base_synonyms)
            elif word.endswith('s'):
                synonyms.update(s + 's' for s in base_synonyms)

        # Filter and sort synonyms
        filtered_synonyms = [s for s in synonyms if self.is_valid_word(s)]
        return sorted(list(filtered_synonyms))

    def is_valid_word(self, word: str) -> bool:
        """Validate word with contextual awareness"""
        return (
                (word.isalnum() or  # Allow alphanumeric
                 (word.replace("'", "").isalnum() and "'" in word) or  # Allow contractions
                 (word.replace("-", "").isalnum() and "-" in word))  # Allow hyphenated words
                and len(word) > 1  # At least 2 characters
                and not word.lower() in self.stop_words  # Not a stop word
        )

    def generate_multi_word_perturbations(self, sentence: str, max_words: int = 4,
                                          max_perturbations: int = 2000) -> List[Tuple[List[int], str]]:
        """Generate perturbations while maintaining context"""
        tokens = self.split_into_words(sentence)

        # Get base words for checking valid indices
        base_tokens = [self.get_base_word(word) for word in tokens]
        valid_indices = [i for i, word in enumerate(base_tokens) if self.is_valid_word(word)]

        perturbations = []
        seen_perturbations = set()

        for num_words in range(1, max_words + 1):
            for word_positions in itertools.combinations(valid_indices, num_words):
                # Get synonyms for base words
                word_synonyms = []
                for pos in word_positions:
                    base_word = base_tokens[pos]
                    synonyms = self.get_enhanced_synonyms(base_word, sentence)

                    # If original word had possessive, add it to synonyms
                    if tokens[pos].endswith("'s"):
                        synonyms = {syn + "'s" for syn in synonyms}

                    word_synonyms.append(list(synonyms))

                if any(not syns for syns in word_synonyms):
                    continue

                for synonym_combination in itertools.product(*word_synonyms):
                    new_tokens = tokens.copy()
                    for pos, synonym in zip(word_positions, synonym_combination):
                        new_tokens[pos] = synonym

                    perturbed = self.reconstruct_sentence(new_tokens)

                    if perturbed != sentence and perturbed not in seen_perturbations:
                        perturbations.append((list(word_positions), perturbed))
                        seen_perturbations.add(perturbed)

                        if len(perturbations) >= max_perturbations:
                            return perturbations

        return perturbations

    def split_into_words(self, sentence: str) -> List[str]:
        """Split sentence while preserving formatting"""
        words = []
        current_word = ""

        for char in sentence:
            if char.isalnum() or char in "'-":
                current_word += char
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                if not char.isspace():
                    words.append(char)

        if current_word:
            words.append(current_word)

        return words

    def reconstruct_sentence(self, tokens: List[str]) -> str:
        """Reconstruct sentence preserving original formatting"""
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


def process_sst2_sentence(sentence: str, output_file: str, max_perturbations: int = 5000):
    """Process a single sentence with context-aware perturbations"""
    try:
        perturbator = ContextAwarePerturbator()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Original: {sentence}\n\n")

            perturbations = perturbator.generate_multi_word_perturbations(
                sentence,
                max_words=4,
                max_perturbations=max_perturbations
            )

            for positions, perturbed in perturbations:
                pos_str = ','.join(str(p + 1) for p in positions)
                f.write(f"Modified {pos_str}: {perturbed}\n")

            print(f"Generated {len(perturbations)} perturbations")

    except Exception as e:
        print(f"Error processing sentence: {str(e)}")
        raise


def create_filename(idx):
    """Create filename using the dataset index"""
    return f"sentence_{idx}.txt"


def process_sst2_dataset():
    """Process all sentences in the SST2 dataset"""
    # Create output directory if it doesn't exist
    output_dir = "full_modified_sentences_synonyms"
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    try:
        dataset = load_dataset("sst2")
        test_set = dataset["test"]
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    total_sentences = len(test_set)
    print(f"Processing {total_sentences} sentences...")

    # Process each sentence
    for idx, example in enumerate(test_set):
        try:
            sentence = example['sentence']

            # Create filename using the dataset index
            filename = create_filename(idx)
            output_file = os.path.join(output_dir, filename)

            # Process the sentence
            process_sst2_sentence(sentence, output_file,max_perturbations=1000)

            # Print progress
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{total_sentences} sentences...")

        except Exception as e:
            print(f"Error processing sentence {idx}: {str(e)}")
            print(f"Problematic sentence: {sentence}")
            continue

    print("Processing complete!")

# Example usage
if __name__ == "__main__":
    # example_sentence = "this film's relationship to actual tension is the same as what christmas-tree flocking in a spray can is to actual snow : a poor -- if durable -- imitation."
    # process_sst2_sentence(example_sentence, "example_output.txt", max_perturbations=5000)
    process_sst2_dataset()