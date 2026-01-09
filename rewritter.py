import spacy
import subprocess
import sys
import streamlit as st
import random
import re
from dataclasses import dataclass
import textstat

@dataclass
class ParaphraseConfig:
    intensity: float = 0.8
    use_passive: bool = True
    restructure: bool = True
    academic_tone: bool = True

@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.info("Downloading language model... This may take a minute.")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        nlp = spacy.load("en_core_web_sm")
    return nlp

class LightParaphraser:
    def __init__(self, nlp):
        self.nlp = nlp
        
        self.synonyms = {
            "show": ["demonstrate", "illustrate", "reveal", "indicate", "display"],
            "find": ["discover", "observe", "identify", "detect", "determine"],
            "suggest": ["imply", "indicate", "propose", "recommend"],
            "increase": ["rise", "elevate", "enhance", "boost", "augment"],
            "decrease": ["reduce", "decline", "diminish", "lessen"],
            "important": ["significant", "crucial", "vital", "essential", "critical"],
            "use": ["utilize", "employ", "apply", "implement"],
            "study": ["research", "investigation", "analysis", "examination"],
            "result": ["outcome", "finding", "consequence", "effect"],
            "method": ["approach", "technique", "procedure", "strategy"],
        }
        
        self.templates = [
            "It has been observed that {clause}",
            "Evidence suggests that {clause}",
            "The findings indicate that {clause}",
            "Research demonstrates that {clause}",
            "It can be seen that {clause}",
        ]
        
        self.connectors = ["Furthermore,", "Moreover,", "However,", "Therefore,", "In addition,"]

    def replace_synonyms(self, sentence: str, intensity: float) -> str:
        doc = self.nlp(sentence)
        tokens = list(doc)
        replaced = 0
        max_replace = int(len(tokens) * intensity)
        
        new_tokens = []
        for token in tokens:
            if replaced >= max_replace:
                new_tokens.append(token.text_with_ws)
                continue
            word = token.text.lower().rstrip(".,!?;:")
            if word in self.synonyms and token.pos_ in ("VERB", "NOUN", "ADJ", "ADV"):
                new_word = random.choice(self.synonyms[word])
                if token.text[0].isupper():
                    new_word = new_word.capitalize()
                suffix = token.text[len(token.text) - len(token.text.lstrip(".,!?;:")):]
                new_tokens.append(new_word + suffix + token.whitespace_)
                replaced += 1
            else:
                new_tokens.append(token.text_with_ws)
        
        return "".join(new_tokens)

    def to_passive(self, sentence: str) -> str:
        doc = self.nlp(sentence)
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subjects = [w for w in token.children if w.dep_ == "nsubj"]
                objects = [w for w in token.children if w.dep_ in ("dobj", "pobj", "attr")]
                if subjects and objects:
                    return f"{objects[0].text.capitalize()} was {token.lemma_}ed by {subjects[0].text}."
        return sentence

    def restructure(self, sentence: str) -> str:
        if random.random() < 0.5:
            clause = re.sub(r'[.?!]$', '', sentence).strip().lower()
            return random.choice(self.templates).format(clause=clause).capitalize() + "."
        return sentence

    def add_tone(self, sentence: str) -> str:
        if random.random() < 0.3 and len(sentence.split()) > 8:
            return f"{random.choice(self.connectors)} {sentence}"
        return sentence

    def paraphrase_sentence(self, sentence: str, config: ParaphraseConfig) -> str:
        if len(sentence.split()) < 5:
            return sentence
        s = self.replace_synonyms(sentence, config.intensity)
        if config.use_passive and random.random() < 0.6:
            s = self.to_passive(s)
        if config.restructure and random.random() < 0.7:
            s = self.restructure(s)
        if config.academic_tone:
            s = self.add_tone(s)
        return s[0].upper() + s[1:] if s else s

    def paraphrase_text(self, text: str, config: ParaphraseConfig) -> dict:
        doc = self.nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        paraphrased = [self.paraphrase_sentence(s, config) for s in sentences]
        
        if len(paraphrased) > 3 and random.random() < 0.4:
            middle = paraphrased[1:-1]
            random.shuffle(middle)
            paraphrased = [paraphrased[0]] + middle + [paraphrased[-1]]
        
        new_text = " ".join(paraphrased)
        
        orig_words = set(re.findall(r'\w+', text.lower()))
        new_words = set(re.findall(r'\w+', new_text.lower()))
        change_pct = round((1 - len(orig_words & new_words) / len(orig_words)) * 100, 1) if orig_words else 0
        
        risk = "Low" if change_pct > 50 else "Medium" if change_pct > 30 else "High"
        
        return {"paraphrased_text": new_text, "word_change_pct": change_pct, "plagiarism_risk": risk}

def main():
    st.set_page_config(page_title="Anti-Plagiarism Rewriter", page_icon="🛡️", layout="wide")
    st.title("🛡️ Anti-Plagiarism Text Rewriter")
    st.markdown("Rule-based paraphrasing to reduce similarity while preserving meaning.")

    with st.sidebar:
        st.header("Settings")
        intensity = st.slider("Intensity", 0.4, 1.0, 0.8, 0.1)
        use_passive = st.checkbox("Use Passive Voice", True)
        restructure = st.checkbox("Restructure Sentences", True)
        academic_tone = st.checkbox("Academic Tone", True)

    nlp = load_models()
    engine = LightParaphraser(nlp)

    text_input = st.text_area("Paste your text:", height=200)

    if st.button("Rewrite Text", type="primary"):
        if text_input.strip():
            config = ParaphraseConfig(intensity, use_passive, restructure, academic_tone)
            with st.spinner("Rewriting..."):
                result = engine.paraphrase_text(text_input, config)
            col1, col2 = st.columns(2)
            with col1: st.subheader("Original"); st.write(text_input)
            with col2: st.subheader("Rewritten"); st.write(result["paraphrased_text"])
            st.success(f"Word change: {result['word_change_pct']}% → Risk: {result['plagiarism_risk']}")
        else:
            st.warning("Enter text to rewrite.")

if __name__ == "__main__":
    main()


