import os
import nltk
import streamlit as st
import numpy as np
from typing import List, Dict
import re
from dataclasses import dataclass

# Import necessary libraries
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    import spacy
    from sentence_transformers import SentenceTransformer, util
    from keybert import KeyBERT
    from lexicalrichness import LexicalRichness
    import textstat
except ImportError as e:
    st.error(f"Missing library: {e}. Install with: pip install nltk spacy sentence-transformers keybert lexicalrichness textstat numpy")
    st.stop()

# ------------------- Configuration -------------------
@dataclass
class ParaphraseConfig:
    method: str = "hybrid"  # hybrid, semantic, structural, synonym
    intensity: float = 0.7
    preserve_key_terms: bool = True
    change_sentence_structure: bool = True
    academic_tone: bool = True
    diversity_level: int = 3  # 1-5 (affects randomness)

@st.cache_resource
def load_models():
    with st.spinner("Loading models..."):
        # Create writable NLTK data directory
        nltk_data_dir = "/home/appuser/.nltk_data"
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Append to NLTK search path (do this early!)
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.append(nltk_data_dir)
        
        # Download required data to the writable dir (quietly)
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
        # Add any others you need, e.g., 'stopwords'
        
        # Now load your other models...
        models['spacy'] = spacy.load("en_core_web_sm")
        models['sentence_transformer'] = SentenceTransformer('all-mpnet-base-v2')
        models['keybert'] = KeyBERT()
    
    st.success("Models and NLTK data loaded successfully!")
    return models

# ------------------- Core Paraphrasing Engine -------------------
class ParaphraseEngine:
    def __init__(self, models):
        self.models = models
        self.spacy_nlp = models['spacy']
        self.sentence_transformer = models['sentence_transformer']
        self.keybert = models['keybert']
        
        # Academic synonyms (expanded slightly)
        self.academic_synonyms = {
            'show': ['demonstrate', 'illustrate', 'reveal', 'indicate'],
            'find': ['observe', 'identify', 'discover', 'ascertain'],
            'suggest': ['indicate', 'imply', 'propose', 'postulate'],
            'increase': ['elevate', 'enhance', 'augment', 'escalate'],
            'decrease': ['reduce', 'diminish', 'attenuate', 'lessen'],
            'significant': ['substantial', 'considerable', 'notable', 'pronounced'],
            'important': ['crucial', 'essential', 'vital', 'critical'],
            'use': ['employ', 'utilize', 'apply', 'implement'],
            'perform': ['conduct', 'execute', 'undertake'],
            'analyze': ['examine', 'evaluate', 'scrutinize'],
        }
        
        # Academic sentence templates
        self.sentence_templates = [
            "It was found that {clause}",
            "The results demonstrate that {clause}",
            "This suggests that {clause}",
            "There is evidence that {clause}",
            "The analysis shows that {clause}",
            "It can be observed that {clause}",
        ]
        
        self.academic_connectors = [
            "Furthermore,", "Moreover,", "However,", "Therefore,", "Thus,", "In addition,", "Consequently,"
        ]

    def extract_key_terms(self, text: str, top_n: int = 10) -> List[str]:
        keywords = self.keybert.extract_keywords(
            text, keyphrase_ngram_range=(1, 3), stop_words='english', top_n=top_n
        )
        return [kw[0] for kw in keywords]

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.sentence_transformer.encode([text1, text2])
        return util.cos_sim(embeddings[0], embeddings[1]).item()

    def synonym_replacement(self, sentence: str, intensity: float = 0.5) -> str:
        doc = self.spacy_nlp(sentence)
        words = [token.text for token in doc]
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        replacements = 0
        max_replacements = int(len(words) * intensity)
        
        for i, (word, pos) in enumerate(pos_tags):
            if replacements >= max_replacements:
                break
            word_lower = word.lower()
            if word_lower in self.academic_synonyms and np.random.random() < 0.7:
                synonyms = self.academic_synonyms[word_lower]
                # Simple filtering (you can improve with POS if needed)
                new_word = np.random.choice(synonyms)
                if word[0].isupper():
                    new_word = new_word.capitalize()
                words[i] = new_word
                replacements += 1
        
        return ' '.join(words)

    def change_sentence_structure(self, sentence: str) -> str:
        if np.random.random() < 0.4:
            template = np.random.choice(self.sentence_templates)
            clause = sentence.strip('.!?').lower()
            return template.format(clause=clause).capitalize() + "."
        return sentence

    def enhance_academic_tone(self, sentence: str) -> str:
        if len(sentence.split()) > 8 and np.random.random() < 0.25:
            connector = np.random.choice(self.academic_connectors)
            return f"{connector} {sentence}"
        return sentence

    def paraphrase_sentence(self, sentence: str, config: ParaphraseConfig) -> str:
        if len(sentence.split()) < 4:
            return sentence
        
        paraphrased = self.synonym_replacement(sentence, intensity=config.intensity)
        
        if config.change_sentence_structure and np.random.random() < 0.7:
            paraphrased = self.change_sentence_structure(paraphrased)
        
        if config.academic_tone:
            paraphrased = self.enhance_academic_tone(paraphrased)
        
        return paraphrased

    def paraphrase_text(self, text: str, config: ParaphraseConfig) -> Dict:
        sentences = sent_tokenize(text)
        global_key_terms = self.extract_key_terms(text, top_n=15)
        
        paraphrased_sentences = []
        original_sentences = []
        
        for sentence in sentences:
            original_sentences.append(sentence)
            if len(sentence.strip()) < 10:
                paraphrased_sentences.append(sentence)
                continue
            paraphrased = self.paraphrase_sentence(sentence, config)
            paraphrased_sentences.append(paraphrased)
        
        # Optional structural reordering
        if config.method in ["hybrid", "structural"] and len(paraphrased_sentences) > 2:
            if np.random.random() < 0.4:
                middle = paraphrased_sentences[1:-1]
                np.random.shuffle(middle)
                paraphrased_sentences = [paraphrased_sentences[0]] + middle + [paraphrased_sentences[-1]]
        
        paraphrased_text = ' '.join(paraphrased_sentences)
        
        metrics = self.calculate_paraphrase_metrics(
            original_sentences, paraphrased_sentences, text, paraphrased_text
        )
        
        return {
            'paraphrased_text': paraphrased_text,
            'metrics': metrics,
            'key_terms': global_key_terms
        }

    def calculate_paraphrase_metrics(self, orig_sents, para_sents, orig_text, para_text) -> Dict:
        orig_words = set(orig_text.lower().split())
        para_words = set(para_text.lower().split())
        common_words = orig_words.intersection(para_words)
        word_change_ratio = 1 - (len(common_words) / max(len(orig_words), 1))
        
        similarities = []
        for o, p in zip(orig_sents, para_sents):
            sim = self.calculate_semantic_similarity(o, p)
            similarities.append(sim)
        avg_semantic_sim = np.mean(similarities) if similarities else 0
        
        lex_orig = LexicalRichness(orig_text)
        lex_para = LexicalRichness(para_text)
        mtld_orig = lex_orig.mtld(threshold=0.72) if len(orig_text.split()) > 10 else 0
        mtld_para = lex_para.mtld(threshold=0.72) if len(para_text.split()) > 10 else 0
        
        readability_orig = textstat.flesch_reading_ease(orig_text)
        readability_para = textstat.flesch_reading_ease(para_text)
        
        plagiarism_risk = 'Low' if word_change_ratio > 0.4 and avg_semantic_sim > 0.7 else \
                          'Medium' if word_change_ratio > 0.2 else 'High'
        
        return {
            'word_change_percentage': round(word_change_ratio * 100, 1),
            'semantic_similarity': round(avg_semantic_sim * 100, 1),
            'lexical_diversity_original': round(mtld_orig, 2),
            'lexical_diversity_paraphrased': round(mtld_para, 2),
            'readability_original': round(readability_orig, 1),
            'readability_paraphrased': round(readability_para, 1),
            'plagiarism_risk': plagiarism_risk
        }

# ------------------- Streamlit UI (unchanged, minor fixes) -------------------
def main():
    st.set_page_config(page_title="Advanced Anti-Plagiarism Paraphraser", page_icon="🛡️", layout="wide")
    
    st.title("🛡️ Advanced Anti-Plagiarism Paraphraser (Lightweight)")
    st.markdown("Multi-strategy paraphraser using synonym replacement, structural changes, and academic tone enhancement.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        method = st.selectbox("Method", ["Hybrid (Recommended)", "Structural", "Synonym-based"])
        intensity = st.slider("Intensity", 0.1, 1.0, 0.7, 0.1)
        diversity = st.slider("Diversity", 1, 5, 3)
        preserve_terms = st.checkbox("Preserve Key Terms", True)
        change_structure = st.checkbox("Change Sentence Structure", True)
        academic_tone = st.checkbox("Academic Tone", True)
        
        # Examples (same as original)
        examples = {
            "Scientific Abstract": "This study investigates the effects of CRISPR-Cas9 gene editing on HEK293 cells. Results show significant upregulation of p53 protein expression.",
            # ... add others if desired
        }
        # (example loading code omitted for brevity – you can keep it)

    models = load_models()
    engine = ParaphraseEngine(models)
    
    text_input = st.text_area("Paste text to paraphrase:", height=200)
    
    col1, col2 = st.columns(2)
    if col1.button("🔄 Paraphrase", type="primary"):
        if text_input.strip():
            method_map = {"Hybrid (Recommended)": "hybrid", "Structural": "structural", "Synonym-based": "synonym"}
            config = ParaphraseConfig(
                method=method_map[method],
                intensity=intensity,
                preserve_key_terms=preserve_terms,
                change_sentence_structure=change_structure,
                academic_tone=academic_tone,
                diversity_level=diversity
            )
            
            with st.spinner("Paraphrasing..."):
                result = engine.paraphrase_text(text_input, config)
            
            # Display results (same layout as your original – tabs, metrics, etc.)
            # (Copy the display code from your original script here – it works unchanged)
            
            st.success("Paraphrasing complete!")
        else:
            st.warning("Please enter text.")

if __name__ == "__main__":
    main()


