
import spacy

class CommonValidator:
    def __init__(self):
        self.temp = []
        """Load the spaCy model for NLP processing."""
        self.nlp = spacy.load("en_core_web_sm")
    
    def is_present_tense(self, doc):
        # 1. Present simple (VBZ/VBP)
        if any(token.tag_ in {"VBZ", "VBP"} for token in doc):
            return True

        # 2. Present continuous (AUX "be" + VBG)
        if any(t.pos_ == "AUX" and t.lemma_ == "be" for t in doc) and any(t.tag_ == "VBG" for t in doc):
            return True

        # 3. Imperative (first verb is VB/VBP)
        first_verb = next((t for t in doc if t.pos_ == "VERB"), None)
        if first_verb and first_verb.tag_ in {"VB", "VBP"}:
            return True

        # 4. Modal + VBG (e.g., "must be running")
        if any(t.tag_ == "MD" for t in doc) and any(t.tag_ == "VBG" for t in doc):
            return True

        # 5. Modal + BE + VBN (e.g., "must be configured")
        if any(t.tag_ == "MD" for t in doc) and any(t.lemma_ == "be" and t.pos_ == "AUX" for t in doc) and any(t.tag_ == "VBN" for t in doc):
            return True

        # 6. Modal + BE + Prep phrase (e.g., "must be in X mode")
        if any(t.tag_ == "MD" for t in doc) and any(t.lemma_ == "be" for t in doc) and any(t.dep_ == "prep" or t.pos_ == "ADP" for t in doc):
            return True

        return False