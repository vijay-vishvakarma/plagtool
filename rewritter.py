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
        nlp = spacy.load("en_core_web_sm")
    st.success("Model loaded – ready to rewrite!")
    return nlp

class LightParaphraser:
    def __init__(self, nlp):
        self.nlp = nlp
        
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
                # Preserve trailing whitespace/punctuation
                suffix = token.text[len(token.text.lstrip(".,!?;:")):]
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
                objects = [w for w in token.children if w.dep_ in ("dobj", "dative", "attr", "pobj")]
                if subjects and objects:
                    obj_text = " ".join([t.text for t in objects[0].subtree])
                    subj_text = subjects[0].text
                    verb = token.lemma_
                    return f"{obj_text.capitalize()} was {verb}ed by {subj_text}."
        return sentence

    def restructure(self, sentence: str) -> str:
        if random.random() < 0.5:
            clause = re.sub(r'[.?!]$', '', sentence).strip().lower()
            template = random.choice(self.templates)
            return template.format(clause=clause).capitalize() + "."
        return sentence

    def add_tone(self, sentence: str) -> str:
        if random.random() < 0.3 and len(sentence.split()) > 8:  # ← Fixed: added missing colon
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
            s = self.restructure(s)
        
        if config.academic_tone:
            s = self.add_tone(s)
        
        # Ensure proper capitalization
        if s and s[0].islower():
            s = s[0].upper() + s[1:]
        return s

    def paraphrase_text(self, text: str, config: ParaphraseConfig) -> dict:
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        paraphrased = [self.paraphrase_sentence(s, config) for s in sentences]
        
        if len(paraphrased) > 3 and random.random() < 0.4:
            middle = paraphrased[1:-1]
            random.shuffle(middle)
            paraphrased = [paraphrased[0]] + middle + [paraphrased[-1]]
        
        new_text = " ".join(paraphrased)
        
        orig_words = set(re.findall(r'\w+', text.lower()))
        new_words = set(re.findall(r'\w+', new_text.lower()))
        overlap = len(orig_words & new_words) / len(orig_words) if orig_words else 0
        change_pct = round((1 - overlap) * 100, 1)
        
        risk = "Low" if change_pct > 50 else "Medium" if change_pct > 30 else "High"
        
        return {
            "paraphrased_text": new_text,
            "word_change_pct": change_pct,
            "plagiarism_risk": risk
        }

def main():
    st.set_page_config(page_title="Ultra-Light Anti-Plagiarism Rewriter", page_icon="🛡️", layout="wide")
    
    st.title("🛡️ Ultra-Light Anti-Plagiarism Rewriter")
    st.markdown("Fast, rule-based rewriting to help avoid plagiarism detection.")

    with st.sidebar:
        st.header("⚙️ Settings")
        intensity = st.slider("Intensity", 0.4, 1.0, 0.8, 0.1)
        use_passive = st.checkbox("Active → Passive Voice", True)
        restructure = st.checkbox("Template Restructuring", True)
        academic_tone = st.checkbox("Academic Tone", True)

    nlp = load_models()
    engine = LightParaphraser(nlp)

    text_input = st.text_area("Paste text to rewrite:", height=220, placeholder="Enter your text here...")

    if st.button("🔄 Rewrite to Avoid Plagiarism", type="primary"):
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
                st.code(result["paraphrased_text"], language="text")
            
            st.success(f"✅ Word change: **{result['word_change_pct']}%** | Risk: **{result['plagiarism_risk']}**")
            
            if result["word_change_pct"] < 40:
                st.info("💡 Try increasing intensity or enabling more options for stronger rewriting.")
        else:
            st.warning("Please enter some text first.")

    st.caption("Best practice: Always review and edit the output manually.")

if __name__ == "__main__":
    main()
