import numpy as np
from typing import List, Dict
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer

from app.helpers.embedding import Embeddings
from app.helpers.preprocess import preprocess
from app.core.logging import setup_logging

logger = setup_logging()


class Summarizer:
    def __init__(self):
        logger.info("Loading Embedding...")
        self.word_vec = Embeddings().load_vector()
        self.key_to_index = self.word_vec.key_to_index
        self.vector_size = self.word_vec.vector_size
        logger.info("Embedding loaded successfully.")

    def generate_centroid_tfidf(self, sentences: List[str]) -> np.ndarray:
        """
        Generates the TF-IDF centroid of important words in the given sentences.

        Args:
            sentences (List[str]): List of sentences to analyze.

        Returns:
            np.ndarray: The centroid vector of the important words.
        """
        tf = TfidfVectorizer()
        tfidf_matrix = tf.fit_transform(sentences).toarray()
        tfidf_sum = np.sum(tfidf_matrix, axis=0)
        tfidf_max = tfidf_sum.max()
        tfidf_normalized = tfidf_sum / tfidf_max

        words = tf.get_feature_names_out()
        important_terms = [
            word
            for i, word in enumerate(words)
            if word in self.key_to_index and tfidf_normalized[i] >= 0.2
        ]

        if not important_terms:
            print("No important terms found for centroid generation.")
            return np.zeros(self.vector_size)

        centroid = np.mean([self.word_vec[word] for word in important_terms], axis=0)
        return centroid

    def sentence_vectorizer(self, sentences: List[str]) -> Dict[int, np.ndarray]:
        """
        Converts each sentence into a vector by averaging the word embeddings of its words.

        Args:
            sentences (List[str]): List of sentences to convert to vectors.

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping sentence indices to their respective vector representation.
        """
        sentence_vectors = {}
        for i, sentence in enumerate(sentences):
            words = sentence.split()
            sentence_vec = np.zeros(self.vector_size)
            word_count = 0
            for word in words:
                if word in self.key_to_index:
                    sentence_vec = np.add(sentence_vec, self.word_vec[word])
                    word_count += 1
            if word_count > 0:
                sentence_vec /= word_count  # Average the word vectors
            sentence_vectors[i] = sentence_vec
        return sentence_vectors

    def sentence_selection(
        self,
        centroid: np.ndarray,
        sentence_vectors: Dict[int, np.ndarray],
        summary_length: int,
    ) -> List[int]:
        """
        Selects the most important sentences based on their similarity to the centroid.

        Args:
            centroid (np.ndarray): The centroid vector representing important terms.
            sentence_vectors (Dict[int, np.ndarray]): Dictionary of sentence indices and their vector representations.
            summary_length (int): The maximum length of the summary in terms of sentence count.

        Returns:
            List[int]: List of selected sentence indices sorted in order of selection.
        """
        if not any(np.linalg.norm(vec) > 0 for vec in sentence_vectors.values()):
            print("No valid sentence vectors found.")
            return []

        similarities = [
            (sentence_id, 1 - cosine(centroid, vec))
            for sentence_id, vec in sentence_vectors.items()
            if np.linalg.norm(vec) > 0
        ]

        # Sort by similarity (descending order) and select top sentences
        ranked_sentences = sorted(similarities, key=lambda x: x[1], reverse=True)

        selected_sentences = []
        total_length = 0
        for sentence_id, _ in ranked_sentences:
            selected_sentences.append(sentence_id)
            total_length += len(sentence_vectors[sentence_id])
            if total_length >= summary_length:
                break

        return sorted(selected_sentences)

    def combine_sentence(
        self, selected_sentence_ids: List[int], sentences: List[str]
    ) -> str:
        """
        Combines the selected sentences into a final summary.

        Args:
            selected_sentence_ids (List[int]): List of sentence indices to be included in the summary.
            sentences (List[str]): List of sentences from the original text.

        Returns:
            str: The combined summary as a string.
        """
        return "। ".join([sentences[i] for i in selected_sentence_ids])

    def show_summary(self, text: str, length_sentence_predict: float = 0.01) -> str:
        """
        Main function to generate a summary from the input text.

        Args:
            text (str): The input text to summarize.
            length_sentence_predict (float): Desired length of the summary as a proportion
                                             of the total text length (default is 0.01).

        Returns:
            str: The generated summary.
        """
        sentences = preprocess(text)

        # Calculate the total length of the text in characters
        total_text_length = len(text)

        if total_text_length < 500:
            length_sentence_predict = max(0.8, length_sentence_predict)
        elif total_text_length < 1500:
            length_sentence_predict = max(0.5, length_sentence_predict)
        else:
            length_sentence_predict = max(0.2, length_sentence_predict)

        # Generate centroid of important words and sentence vectors
        centroid = self.generate_centroid_tfidf(sentences)
        sentence_vectors = self.sentence_vectorizer(sentences)

        # Determine summary length (either by character limit or number of sentences)
        if length_sentence_predict < 1:
            total_char_count = sum(len(sentence) for sentence in sentences)
            length_sentence_predict = int(total_char_count * length_sentence_predict)

        # Select important sentences based on centroid similarity
        selected_sentence_ids = self.sentence_selection(
            centroid, sentence_vectors, length_sentence_predict
        )

        return f"{self.combine_sentence(selected_sentence_ids, sentences)}।"
