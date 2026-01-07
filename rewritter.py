import os
import streamlit as st
import numpy as np
import random
import re
from dataclasses import dataclass

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    import spacy
    from lexicalrichness import LexicalRichness
    import textstat
except ImportError as e:
    st.error(f"Missing library: {e}. Install with: pip install nltk spacy lexicalrichness textstat")
    st.stop()

@dataclass
class ParaphraseConfig:
    intensity: float = 0.8          # How many words to replace
    use_passive: bool = True        # Active <-> Passive conversion
    restructure: bool = True        # Use templates & reordering
    academic_tone: bool = True      # Add connectors/hedging

@st.cache_resource
def load_models():
    # NLTK data (writable path for Streamlit Cloud)
    nltk_data_dir = "/home/appuser/.nltk_data"
    os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
    
    # SpaCy small model (for dependency parsing in passive conversion)
    nlp = spacy.load("en_core_web_sm")
    
    st.success("Lightweight models loaded!")
    return {"spacy": nlp}

class RuleBasedParaphraser:
    def __init__(self, nlp):
        self.nlp = nlp
        
        # Large synonym database (expand as needed)
        self.synonyms = {
            "show": ["demonstrate", "illustrate", "reveal", "indicate", "display"],
            "find": ["discover", "observe", "identify", "detect", "determine"],
            "suggest": ["imply", "indicate", "propose", "recommend", "hint"],
            "increase": ["rise", "grow", "elevate", "enhance", "boost"],
            "decrease": ["fall", "reduce", "decline", "diminish", "drop"],
            "important": ["significant", "crucial", "vital", "essential", "key"],
            "use": ["utilize", "employ", "apply", "implement"],
            "study": ["research", "investigation", "analysis", "examination"],
            "result": ["outcome", "finding", "consequence", "effect"],
            "method": ["approach", "technique", "procedure", "strategy"],
            # Add more common words...
        }
        
        self.templates = [
            "It has been observed that {clause}",
            "Research indicates that {clause}",
            "Evidence suggests that {clause}",
            "The analysis reveals that {clause}",
            "Findings demonstrate that {clause}",
            "It can be argued that {clause}",
        ]
        
        self.connectors = [
            "Furthermore,", "Moreover,", "However,", "Therefore,", "In addition,", 
            "Consequently,", "Notably,", "Specifically,"
        ]

    def replace_synonyms(self, sentence: str, intensity: float) -> str:
        words = word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        
        replaced = 0
        max_replace = int(len(words) * intensity)
        
        for i, (word, pos) in enumerate(tagged):
            if replaced >= max_replace:
                break
            low = word.lower().rstrip('.,!?')
            if low in self.synonyms and pos.startswith(('NN', 'VB', 'JJ', 'RB')):
                syn = random.choice(self.synonyms[low])
                if word[0].isupper():
                    syn = syn.capitalize()
                words[i] = syn + word[len(low):]  # Preserve punctuation
                replaced += 1
        
        return " ".join(words)

    def to_passive(self, sentence: str) -> str:
        doc = self.nlp(sentence)
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subjects = [w for w in token.children if w.dep_ == "nsubj"]
                objects = [w for w in token.children if w.dep_ in ("dobj", "iobj")]
                if subjects and objects:
                    return f"{objects[0]} was {token.lemma_}ed by {subjects[0]}."
        return sentence

    def restructure_sentence(self, sentence: str) -> str:
        if random.random() < 0.5:
            clause = sentence.strip(".!?").lower()
            template = random.choice(self.templates)
            return template.format(clause=clause).capitalize() + "."
        return sentence

    def add_academic_tone(self, sentence: str) -> str:
        if random.random() < 0.3 and len(sentence.split()) > 10:
            connector = random.choice(self.connectors)
            return f"{connector} {sentence}"
        return sentence

    def paraphrase_sentence(self, sentence: str, config: ParaphraseConfig) -> str:
        if len(sentence.split()) < 5:
            return sentence
        
        s = self.replace_synonyms(sentence, config.intensity)
        
        if config.use_passive and random.random() < 0.6:
            s = self.to_passive(s)
        
        if config.restructure and random.random() < 0.7:
            s = self.restructure_sentence(s)
        
        if config.academic_tone:
            s = self.add_academic_tone(s)
        
        return s.capitalize()

    def paraphrase_text(self, text: str, config: ParaphraseConfig) -> dict:
        sentences = sent_tokenize(text)
        paraphrased = [self.paraphrase_sentence(s, config) for s in sentences]
        
        # Optional: reorder middle sentences for structural change
        if len(paraphrased) > 3 and random.random() < 0.4:
            middle = paraphrased[1:-1]
            random.shuffle(middle)
            paraphrased = [paraphrased[0]] + middle + [paraphrased[-1]]
        
        new_text = " ".join(paraphrased)
        
        # Simple metrics (no heavy models)
        orig_words = set(re.findall(r'\w+', text.lower()))
        new_words = set(re.findall(r'\w+', new_text.lower()))
        word_change = 1 - len(orig_words & new_words) / max(len(orig_words), 1)
        
        return {
            "paraphrased_text": new_text,
            "word_change_pct": round(word_change * 100, 1),
            "plagiarism_risk": "Low" if word_change > 0.5 else "Medium" if word_change > 0.3 else "High"
        }

def main():
    st.set_page_config(page_title="Rule-Based Anti-Plagiarism Rewriter", page_icon="🛡️", layout="wide")
    st.title("🛡️ Rule-Based Anti-Plagiarism Rewriter (Ultra-Light)")
    st.markdown("Advanced synonym replacement, voice conversion, and structural rewriting to minimize plagiarism detection.")

    with st.sidebar:
        st.header("⚙️ Settings")
        intensity = st.slider("Rewriting Intensity", 0.3, 1.0, 0.8, 0.1)
        use_passive = st.checkbox("Active/Passive Conversion", True)
        restructure = st.checkbox("Template Restructuring", True)
        academic_tone = st.checkbox("Academic Connectors", True)

    nlp = load_models()["spacy"]
    engine = RuleBasedParaphraser(nlp)

    text_input = st.text_area("Input text to rewrite:", height=200)

    if st.button("🔄 Rewrite Text", type="primary"):
        if text_input.strip():
            config = ParaphraseConfig(
                intensity=intensity,
                use_passive=use_passive,
                restructure=restructure,
                academic_tone=academic_tone
            )
            with st.spinner("Rewriting..."):
                result = engine.paraphrase_text(text_input, config)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.write(text_input)
            with col2:
                st.subheader("Rewritten")
                st.write(result["paraphrased_text"])
                st.code(result["paraphrased_text"])
            
            st.success(f"Word change: {result['word_change_pct']}% | Risk: {result['plagiarism_risk']}")
        else:
            st.warning("Enter text first.")

if __name__ == "__main__":
    main()
