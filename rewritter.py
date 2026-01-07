import os
import streamlit as st
import random
import re
from dataclasses import dataclass

try:
    import spacy
    from lexicalrichness import LexicalRichness
    import textstat
except ImportError as e:
    st.error(f"Missing library: {e}. Install with: pip install spacy lexicalrichness textstat")
    st.stop()

@dataclass
class ParaphraseConfig:
    intensity: float = 0.8
    use_passive: bool = True
    restructure: bool = True
    academic_tone: bool = True

@st.cache_resource
def load_models():
    with st.spinner("Loading lightweight model..."):
        # Load spaCy small model (pre-installed via requirements.txt)
        nlp = spacy.load("en_core_web_sm")
    st.success("Model loaded – ready to rewrite!")
    return nlp

class LightParaphraser:
    def __init__(self, nlp):
        self.nlp = nlp
        
        # Expanded synonym dictionary (add more as needed)
        self.synonyms = {
            "show": ["demonstrate", "illustrate", "reveal", "indicate", "display", "exhibit"],
            "find": ["discover", "observe", "identify", "detect", "determine", "uncover"],
            "suggest": ["imply", "indicate", "propose", "recommend", "hint at"],
            "increase": ["rise", "grow", "elevate", "enhance", "boost", "augment"],
            "decrease": ["fall", "reduce", "decline", "diminish", "drop", "lessen"],
            "important": ["significant", "crucial", "vital", "essential", "critical", "key"],
            "use": ["utilize", "employ", "apply", "implement", "adopt"],
            "study": ["research", "investigation", "analysis", "examination", "exploration"],
            "result": ["outcome", "finding", "consequence", "effect", "product"],
            "method": ["approach", "technique", "procedure", "strategy", "methodology"],
            "prove": ["confirm", "validate", "establish", "verify"],
            "cause": ["lead to", "trigger", "induce", "produce", "generate"],
            "affect": ["influence", "impact", "alter", "modify"],
            "good": ["effective", "beneficial", "positive", "favorable"],
            "bad": ["detrimental", "negative", "harmful", "adverse"],
        }
        
        self.templates = [
            "It has been observed that {clause}",
            "Evidence suggests that {clause}",
            "The findings indicate that {clause}",
            "Research demonstrates that {clause}",
            "It can be seen that {clause}",
            "The analysis reveals that {clause}",
            "Studies confirm that {clause}",
        ]
        
        self.connectors = [
            "Furthermore,", "Moreover,", "However,", "Therefore,", 
            "In addition,", "Consequently,", "Notably,", "Specifically,"
        ]

    def replace_synonyms(self, sentence: str, intensity: float) -> str:
        doc = self.nlp(sentence)
        tokens = list(doc)
        replaced = 0
        max_replace = int(len(tokens) * intensity)
        
        for token in tokens:
            if replaced >= max_replace:
                break
            word = token.text.lower().strip(".,!?")
            if word in self.synonyms and token.pos_ in ("VERB", "NOUN", "ADJ", "ADV"):
                new_word = random.choice(self.synonyms[word])
                if token.text[0].isupper():
                    new_word = new_word.capitalize()
                token.text = new_word + token.text[len(word):]  # Keep punctuation
                replaced += 1
        
        return "".join([t.text_with_ws for t in doc])

    def to_passive(self, sentence: str) -> str:
        doc = self.nlp(sentence)
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subjects = [w for w in token.children if w.dep_ == "nsubj"]
                objects = [w for w in token.children if w.dep_ in ("dobj", "dative", "attr")]
                if subjects and objects:
                    obj = objects[0].text
                    subj = subjects[0].text
                    verb = token.lemma_
                    return f"{obj.capitalize()} was {verb}ed by {subj}.".capitalize()
        return sentence

    def restructure(self, sentence: str) -> str:
        if random.random() < 0.5:
            clause = re.sub(r'[.?!]$', '', sentence).lower()
            template = random.choice(self.templates)
            return template.format(clause=clause).capitalize() + "."
        return sentence

    def add_tone(self, sentence: str) -> str:
        if random.random() < 0.3 and len(sentence.split()) > 8
