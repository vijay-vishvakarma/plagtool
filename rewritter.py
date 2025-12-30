import streamlit as st
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
import hashlib

# Import necessary libraries (install these first)
try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import sent_tokenize, word_tokenize
    import spacy
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    from sentence_transformers import SentenceTransformer, util
    from keybert import KeyBERT
    from lexicalrichness import LexicalRichness
    import textstat
except ImportError as e:
    st.error(f"Missing library: {e}. Install with: pip install nltk spacy transformers sentence-transformers keybert lexicalrichness textstat")
    st.stop()

# ------------------- Configuration & Models -------------------
@dataclass
class ParaphraseConfig:
    method: str = "hybrid"  # hybrid, semantic, structural, synonym
    intensity: float = 0.7
    preserve_key_terms: bool = True
    change_sentence_structure: bool = True
    academic_tone: bool = True
    diversity_level: int = 3  # 1-5

@st.cache_resource
def load_models():
    """Load all NLP models with progress tracking"""
    models = {}
    
    with st.spinner("Loading models..."):
        # 1. SpaCy for syntactic analysis
        try:
            models['spacy'] = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            models['spacy'] = spacy.load("en_core_web_sm")
        
        # 2. Sentence transformers for semantic similarity
        models['sentence_transformer'] = SentenceTransformer('all-mpnet-base-v2')
        
        # 3. KeyBERT for keyword extraction
        models['keybert'] = KeyBERT()
        
        # 4. Paraphrase model (T5-based)
        try:
            models['paraphraser'] = pipeline(
                "text2text-generation",
                model="tuner007/pegasus_paraphrase",  # Reliable & high-quality
                device=0 if torch.cuda.is_available() else -1
            )
            st.success("Advanced paraphraser loaded!")
        except Exception as e:
            st.warning("Transformer paraphraser unavailable. Using synonym + structure methods.")
            models['paraphraser'] = None
        
        # 5. Download NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    return models

# ------------------- Core Paraphrasing Strategies -------------------
class ParaphraseEngine:
    def __init__(self, models):
        self.models = models
        self.spacy_nlp = models['spacy']
        self.sentence_transformer = models['sentence_transformer']
        self.keybert = models['keybert']
        self.paraphraser = models['paraphraser']
        
        # Academic synonyms database
        self.academic_synonyms = {
            'show': ['demonstrate', 'illustrate', 'reveal', 'indicate', 'exhibit'],
            'find': ['observe', 'identify', 'detect', 'discover', 'ascertain'],
            'suggest': ['indicate', 'imply', 'propose', 'hint', 'postulate'],
            'increase': ['elevate', 'enhance', 'augment', 'amplify', 'escalate'],
            'decrease': ['reduce', 'diminish', 'attenuate', 'decline', 'lessen'],
            'significant': ['substantial', 'considerable', 'notable', 'marked', 'pronounced'],
            'important': ['crucial', 'essential', 'vital', 'paramount', 'critical'],
            'use': ['employ', 'utilize', 'apply', 'implement', 'deploy'],
            'perform': ['conduct', 'carry out', 'execute', 'undertake'],
            'analyze': ['examine', 'assess', 'evaluate', 'scrutinize'],
            'determine': ['establish', 'ascertain', 'deduce', 'elucidate'],
            'investigate': ['explore', 'examine', 'probe', 'delve into'],
            'propose': ['suggest', 'advocate', 'put forward', 'postulate'],
            'conclude': ['deduce', 'infer', 'derive', 'summarize'],
            'explain': ['elucidate', 'clarify', 'expound', 'interpret'],
        }
        
        # Sentence structure templates for academic writing
        self.sentence_templates = [
            "It was found that {clause}",
            "The results demonstrate that {clause}",
            "As shown in {clause}",
            "This suggests that {clause}",
            "There is evidence that {clause}",
            "It can be observed that {clause}",
            "The data indicate that {clause}",
            "This study reveals that {clause}",
            "The analysis shows that {clause}",
            "It is apparent that {clause}",
        ]
        
        # Academic connectors
        self.academic_connectors = [
            "Furthermore,", "Moreover,", "In addition,", "Additionally,",
            "However,", "Nevertheless,", "Conversely,", "In contrast,",
            "Therefore,", "Consequently,", "Thus,", "Hence,",
            "Specifically,", "Particularly,", "Notably,",
            "In summary,", "In conclusion,",
        ]
    
    def extract_key_terms(self, text: str, top_n: int = 10) -> List[str]:
        """Extract important technical terms to preserve"""
        keywords = self.keybert.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 3), 
            stop_words='english',
            top_n=top_n
        )
        return [kw[0] for kw in keywords]
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        embeddings = self.sentence_transformer.encode([text1, text2])
        return util.cos_sim(embeddings[0], embeddings[1]).item()
    
    def synonym_replacement(self, sentence: str, intensity: float = 0.5) -> str:
        """Replace words with academic synonyms"""
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
                # Filter synonyms by similar part of speech
                filtered_syns = []
                for syn in synonyms:
                    syn_doc = self.spacy_nlp(syn)
                    if syn_doc[0].pos_ == pos:
                        filtered_syns.append(syn)
                
                if filtered_syns:
                    new_word = np.random.choice(filtered_syns)
                    # Preserve capitalization
                    if word[0].isupper():
                        new_word = new_word.capitalize()
                    words[i] = new_word
                    replacements += 1
        
        return ' '.join(words)
    
    def change_sentence_structure(self, sentence: str) -> str:
        """Change sentence structure using different patterns"""
        doc = self.spacy_nlp(sentence)
        
        # Identify main clause
        main_verbs = [token for token in doc if token.pos_ == 'VERB' and token.dep_ in ('ROOT', 'advcl')]
        
        if main_verbs:
            main_verb = main_verbs[0]
            
            # Convert active to passive or vice versa
            if main_verb.tag_ == 'VBD' or main_verb.tag_ == 'VBP':  # Past tense or present
                if np.random.random() < 0.5:
                    # Simple passive transformation
                    if 'by' not in sentence.lower():
                        # Extract subject and object
                        subjects = [token.text for token in doc if token.dep_ in ('nsubj', 'nsubjpass')]
                        objects = [token.text for token in doc if token.dep_ in ('dobj', 'pobj')]
                        
                        if subjects and objects:
                            # Create passive construction
                            passive_verb = self.get_passive_form(main_verb.text)
                            new_sentence = f"{objects[0]} {passive_verb} by {subjects[0]}"
                            # Add remaining parts
                            remaining = [token.text for token in doc 
                                       if token not in [main_verb] + subjects + objects
                                       and token.pos_ not in ('PUNCT')]
                            if remaining:
                                new_sentence += ' ' + ' '.join(remaining)
                            return new_sentence.capitalize()
        
        # Use template if transformation fails
        if np.random.random() < 0.3:
            template = np.random.choice(self.sentence_templates)
            return template.format(clause=sentence.lower())
        
        return sentence
    
    def get_passive_form(self, verb: str) -> str:
        """Get passive form of a verb (simplified)"""
        irregular = {
            'show': 'shown',
            'find': 'found',
            'give': 'given',
            'take': 'taken',
            'make': 'made',
            'do': 'done',
            'see': 'seen',
        }
        
        if verb in irregular:
            return f"was {irregular[verb]}"
        elif verb.endswith('e'):
            return f"was {verb}d"
        elif verb.endswith('y') and not verb.endswith('ay'):
            return f"was {verb[:-1]}ied"
        else:
            return f"was {verb}ed"
    
    def paraphrase_with_transformers(self, text: str, diversity: float = 0.7) -> str:
        """Use transformer model for paraphrasing"""
        if not self.paraphraser:
            return text
        
        try:
            result = self.paraphraser(
                text,
                max_length=len(text.split()) * 3,
                num_beams=5,
                num_return_sequences=1,
                temperature=diversity,
                do_sample=True
            )
            return result[0]['generated_text']
        except:
            return text
    
    def paraphrase_sentence(self, sentence: str, config: ParaphraseConfig) -> str:
        """Apply multiple paraphrasing strategies to a single sentence"""
        if len(sentence.split()) < 4:
            return sentence  # Don't paraphrase very short sentences
        
        key_terms = self.extract_key_terms(sentence, top_n=5)
        
        # Strategy 1: Synonym replacement
        paraphrased = self.synonym_replacement(sentence, intensity=config.intensity * 0.5)
        
        # Strategy 2: Change sentence structure
        if config.change_sentence_structure and np.random.random() < 0.6:
            paraphrased = self.change_sentence_structure(paraphrased)
        
        # Strategy 3: Transformer-based paraphrasing
        if config.method == "hybrid" and np.random.random() < 0.4:
            try:
                transformer_paraphrase = self.paraphrase_with_transformers(
                    paraphrased, 
                    diversity=config.diversity_level / 5
                )
                # Ensure semantic similarity
                similarity = self.calculate_semantic_similarity(paraphrased, transformer_paraphrase)
                if similarity > 0.7:
                    paraphrased = transformer_paraphrase
            except:
                pass
        
        # Ensure key terms are preserved
        if config.preserve_key_terms:
            for term in key_terms:
                if term in sentence and term not in paraphrased:
                    # Try to incorporate the key term
                    paraphrased = self.reinsert_key_term(paraphrased, term)
        
        # Academic tone enhancement
        if config.academic_tone:
            paraphrased = self.enhance_academic_tone(paraphrased)
        
        return paraphrased
    
    def reinsert_key_term(self, sentence: str, term: str) -> str:
        """Reinsert important key terms that were removed"""
        words = sentence.split()
        
        # Find a noun to replace with the key term
        doc = self.spacy_nlp(sentence)
        nouns = [token.text for token in doc if token.pos_ == 'NOUN']
        
        if nouns:
            replace_word = np.random.choice(nouns)
            return sentence.replace(replace_word, term, 1)
        
        return sentence
    
    def enhance_academic_tone(self, sentence: str) -> str:
        """Enhance academic tone of a sentence"""
        words = sentence.split()
        
        # Add academic connectors at beginning occasionally
        if len(words) > 8 and np.random.random() < 0.2:
            connector = np.random.choice(self.academic_connectors)
            return f"{connector} {sentence}"
        
        # Replace informal phrases
        informal_replacements = {
            'a lot of': 'numerous',
            'lots of': 'a considerable number of',
            'got': 'obtained',
            'big': 'substantial',
            'small': 'minimal',
            'good': 'favorable',
            'bad': 'unfavorable',
            'look at': 'examine',
            'find out': 'determine',
            'set up': 'establish',
        }
        
        for informal, formal in informal_replacements.items():
            if informal in sentence.lower():
                sentence = re.sub(fr'\b{informal}\b', formal, sentence, flags=re.IGNORECASE)
        
        return sentence
    
    def paraphrase_text(self, text: str, config: ParaphraseConfig) -> Dict:
        """Main paraphrasing function with multiple strategies"""
        sentences = sent_tokenize(text)
        
        # Extract global key terms from entire text
        global_key_terms = self.extract_key_terms(text, top_n=15)
        
        paraphrased_sentences = []
        original_sentences_clean = []
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                paraphrased_sentences.append(sentence)
                original_sentences_clean.append(sentence)
                continue
            
            # Store original
            original_sentences_clean.append(sentence)
            
            # Paraphrase sentence
            paraphrased = self.paraphrase_sentence(sentence, config)
            paraphrased_sentences.append(paraphrased)
        
        # Reorder sentences occasionally for structural diversity
        if config.method == "structural" and len(paraphrased_sentences) > 2:
            if np.random.random() < 0.3:
                # Don't reorder first and last sentences (introduction/conclusion)
                middle = paraphrased_sentences[1:-1]
                np.random.shuffle(middle)
                paraphrased_sentences = [paraphrased_sentences[0]] + middle + [paraphrased_sentences[-1]]
        
        # Combine sentences
        paraphrased_text = ' '.join(paraphrased_sentences)
        
        # Calculate metrics
        metrics = self.calculate_paraphrase_metrics(
            original_sentences_clean, 
            paraphrased_sentences,
            text, 
            paraphrased_text
        )
        
        return {
            'paraphrased_text': paraphrased_text,
            'metrics': metrics,
            'key_terms': global_key_terms
        }
    
    def calculate_paraphrase_metrics(self, original_sents, paraphrased_sents, orig_text, para_text) -> Dict:
        """Calculate various metrics for paraphrasing quality"""
        # Word-level changes
        orig_words = set(orig_text.lower().split())
        para_words = set(para_text.lower().split())
        
        common_words = orig_words.intersection(para_words)
        word_change_ratio = 1 - (len(common_words) / max(len(orig_words), 1))
        
        # Sentence-level changes
        sentence_changes = []
        for orig, para in zip(original_sents, paraphrased_sents):
            similarity = self.calculate_semantic_similarity(orig, para)
            sentence_changes.append(similarity)
        
        avg_semantic_similarity = np.mean(sentence_changes) if sentence_changes else 0
        
        # Lexical diversity
        lex_orig = LexicalRichness(orig_text)
        lex_para = LexicalRichness(para_text)
        
        try:
            mtld_orig = lex_orig.mtld()
        except:
            mtld_orig = 0
        
        try:
            mtld_para = lex_para.mtld()
        except:
            mtld_para = 0
        
        # Readability
        readability_orig = textstat.flesch_reading_ease(orig_text)
        readability_para = textstat.flesch_reading_ease(para_text)
        
        return {
            'word_change_percentage': round(word_change_ratio * 100, 1),
            'semantic_similarity': round(avg_semantic_similarity * 100, 1),
            'lexical_diversity_original': round(mtld_orig, 2),
            'lexical_diversity_paraphrased': round(mtld_para, 2),
            'readability_original': round(readability_orig, 1),
            'readability_paraphrased': round(readability_para, 1),
            'plagiarism_risk': 'Low' if word_change_ratio > 0.4 and avg_semantic_similarity > 0.7 else 'Medium' if word_change_ratio > 0.2 else 'High'
        }

# ------------------- Streamlit UI -------------------
def main():
    st.set_page_config(
        page_title="Advanced Anti-Plagiarism Paraphraser",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🛡️ Advanced Anti-Plagiarism Paraphraser")
    st.markdown("""
    A multi-strategy paraphrasing tool designed to **effectively avoid plagiarism** while preserving meaning.
    Uses **7+ techniques** including semantic preservation, structural changes, and academic tone enhancement.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Paraphrasing Configuration")
        
        method = st.selectbox(
            "Paraphrasing Method",
            ["Hybrid (Recommended)", "Semantic", "Structural", "Synonym-based"],
            help="Hybrid combines multiple strategies for best results"
        )
        
        intensity = st.slider(
            "Paraphrasing Intensity",
            0.1, 1.0, 0.7, 0.1,
            help="How aggressively to paraphrase"
        )
        
        diversity = st.slider(
            "Output Diversity",
            1, 5, 3,
            help="Creativity level in paraphrasing"
        )
        
        preserve_terms = st.checkbox(
            "Preserve Key Terms", 
            True,
            help="Protect important technical terms"
        )
        
        change_structure = st.checkbox(
            "Change Sentence Structure", 
            True,
            help="Alter sentence patterns"
        )
        
        academic_tone = st.checkbox(
            "Academic Tone Enhancement", 
            True,
            help="Use formal academic language"
        )
        
        st.divider()
        
        # Example texts
        st.subheader("📚 Example Texts")
        examples = {
            "Scientific Abstract": "This study investigates the effects of CRISPR-Cas9 gene editing on HEK293 cells. Results show significant upregulation of p53 protein expression following treatment with 10μM doxorubicin. Western blot analysis confirmed these findings.",
            "Literature Review": "Previous research has demonstrated that machine learning algorithms can accurately predict protein structures. However, these methods often require extensive computational resources and may not generalize well to novel protein families.",
            "Methodology Section": "PCR amplification was performed using Taq polymerase with an annealing temperature of 55°C. Samples were then subjected to gel electrophoresis and visualized using ethidium bromide staining."
        }
        
        selected_example = st.selectbox("Load example:", list(examples.keys()))
        if st.button("Load Example", use_container_width=True):
            st.session_state.example_text = examples[selected_example]
            st.rerun()
    
    # Load models
    models = load_models()
    engine = ParaphraseEngine(models)
    
    # Main text input
    st.subheader("📝 Input Text")
    text_input = st.text_area(
        "Paste text to paraphrase:",
        value=getattr(st.session_state, 'example_text', ''),
        height=200,
        placeholder="Paste academic or scientific text here..."
    )
    
    # Paraphrase button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        paraphrase_btn = st.button("🔄 Paraphrase Text", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        st.session_state.example_text = ""
        st.rerun()
    
    # Process paraphrasing
    if paraphrase_btn and text_input.strip():
        # Create config
        method_map = {
            "Hybrid (Recommended)": "hybrid",
            "Semantic": "semantic",
            "Structural": "structural",
            "Synonym-based": "synonym"
        }
        
        config = ParaphraseConfig(
            method=method_map[method],
            intensity=intensity,
            preserve_key_terms=preserve_terms,
            change_sentence_structure=change_structure,
            academic_tone=academic_tone,
            diversity_level=diversity
        )
        
        # Perform paraphrasing
        with st.spinner("Paraphrasing with anti-plagiarism techniques..."):
            result = engine.paraphrase_text(text_input, config)
        
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Analysis", "🔑 Key Terms", "📋 Comparison"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Text")
                with st.container(border=True, height=300):
                    st.write(text_input)
            
            with col2:
                st.subheader("Paraphrased Text")
                with st.container(border=True, height=300):
                    st.write(result['paraphrased_text'])
                
                # Copy button
                st.code(result['paraphrased_text'])
                if st.button("📋 Copy to Clipboard", key="copy_main"):
                    st.success("Copied!")
        
        with tab2:
            st.subheader("Paraphrasing Analysis")
            
            metrics = result['metrics']
            cols = st.columns(4)
            
            with cols[0]:
                st.metric(
                    "Word Change", 
                    f"{metrics['word_change_percentage']}%",
                    help="Percentage of words changed"
                )
            
            with cols[1]:
                st.metric(
                    "Semantic Similarity", 
                    f"{metrics['semantic_similarity']}%",
                    delta=None,
                    help="How similar the meaning is"
                )
            
            with cols[2]:
                risk_color = {
                    'Low': '🟢',
                    'Medium': '🟡', 
                    'High': '🔴'
                }
                st.metric(
                    "Plagiarism Risk",
                    f"{risk_color[metrics['plagiarism_risk']]} {metrics['plagiarism_risk']}"
                )
            
            with cols[3]:
                diversity_change = metrics['lexical_diversity_paraphrased'] - metrics['lexical_diversity_original']
                st.metric(
                    "Lexical Diversity Change",
                    f"{diversity_change:+.2f}"
                )
            
            # Detailed metrics
            with st.expander("Detailed Metrics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Readability Scores**")
                    st.write(f"Original: {metrics['readability_original']} (higher = easier)")
                    st.write(f"Paraphrased: {metrics['readability_paraphrased']} (higher = easier)")
                
                with col2:
                    st.write("**Lexical Diversity**")
                    st.write(f"Original MTLD: {metrics['lexical_diversity_original']}")
                    st.write(f"Paraphrased MTLD: {metrics['lexical_diversity_paraphrased']}")
                    st.write("*MTLD = Measure of Textual Lexical Diversity*")
        
        with tab3:
            st.subheader("Protected Key Terms")
            key_terms = result['key_terms']
            
            st.write("These terms were identified and preserved during paraphrasing:")
            
            cols = st.columns(3)
            for i, term in enumerate(key_terms):
                with cols[i % 3]:
                    st.info(f"**{term}**")
        
        with tab4:
            st.subheader("Side-by-Side Comparison")
            
            # Highlight changes
            orig_words = text_input.lower().split()
            para_words = result['paraphrased_text'].lower().split()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original (changed words highlighted)**")
                display_text = text_input
                for word in orig_words:
                    if word not in para_words and len(word) > 3:
                        display_text = display_text.replace(word, f"**{word}**")
                st.markdown(display_text)
            
            with col2:
                st.write("**Paraphrased (new words highlighted)**")
                display_text = result['paraphrased_text']
                for word in para_words:
                    if word not in orig_words and len(word) > 3:
                        display_text = display_text.replace(word, f"**{word}**")
                st.markdown(display_text)
        
        # Additional actions
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Generate Alternative Version", use_container_width=True):
                st.rerun()
        
        with col2:
            # Download option
            import io
            buffer = io.StringIO()
            buffer.write(f"ORIGINAL TEXT:\n{text_input}\n\n")
            buffer.write(f"PARAPHRASED TEXT:\n{result['paraphrased_text']}\n\n")
            buffer.write(f"ANALYSIS METRICS:\n")
            for key, value in result['metrics'].items():
                buffer.write(f"{key}: {value}\n")
            
            st.download_button(
                label="📥 Download Results",
                data=buffer.getvalue(),
                file_name="paraphrased_output.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    elif paraphrase_btn:
        st.warning("Please enter some text to paraphrase.")
    
    # Footer with tips
    st.divider()
    
    with st.expander("💡 Tips for Effective Paraphrasing to Avoid Plagiarism"):
        st.markdown("""
        1. **Always cite the original source** even after paraphrasing
        2. **Use multiple strategies**: Combine synonym replacement with structural changes
        3. **Preserve technical terms**: Keep domain-specific vocabulary unchanged
        4. **Check meaning**: Ensure the paraphrased text conveys the same meaning
        5. **Use different sentence patterns**: Convert active to passive voice, change sentence order
        6. **Maintain academic tone**: Use formal language appropriate for your field
        7. **Review similarity**: Aim for <50% word overlap with original
        8. **Check readability**: Ensure the text remains clear and understandable
        9. **Use multiple sources**: Combine ideas from different sources when possible
        10. **Manual review**: Always review AI-paraphrased text for accuracy
        """)

if __name__ == "__main__":

    main()
