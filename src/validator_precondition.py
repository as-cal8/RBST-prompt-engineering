import spacy
from collections import defaultdict

"""
    precondition should be present tense and describe a state or requirement.
    Returns:
        _type_: _description_
"""

class PreconditionValidator:
    def __init__(self):
        """Load the spaCy model for NLP processing."""
        self.nlp = spacy.load("en_core_web_sm")

    def describes_state_or_requirement(self, doc):
        for token in doc:
            if token.lemma_ in {"be", "require", "have", "must", "should", "can", "shall", "exist"}:
                return True
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                # Covers imperative and declarative verbs
                return True
        return False
    
    def is_present_tense(self, doc):
        """
        Checks if the sentence is in present tense or expresses a present-state requirement.
        """
        # 1. Present simple
        if any(token.tag_ in {"VBZ", "VBP"} for token in doc):
            return True

        # 2. Present continuous: AUX (is/are/am) + VBG
        if any(token.pos_ == "AUX" and token.lemma_ == "be" for token in doc) and \
        any(token.tag_ == "VBG" for token in doc):
            return True

        # 3. Imperative (starts with verb)
        if len(doc) > 0 and doc[0].tag_ == "VB":
            return True

        # 4. Modal + VBG (e.g. must be running)
        if any(token.tag_ == "MD" for token in doc) and \
        any(token.tag_ == "VBG" for token in doc):
            return True

        # ✅ 5. Modal + BE + VBN (e.g. must be configured, must be known)
        if any(token.tag_ == "MD" for token in doc) and \
        any(token.lemma_ == "be" and token.pos_ == "AUX" for token in doc) and \
        any(token.tag_ == "VBN" for token in doc):
            return True

        # ✅ 6. Modal + BE + Prep phrase / noun (e.g. "must be in X mode")
        if any(token.tag_ == "MD" for token in doc) and \
        any(token.lemma_ == "be" for token in doc) and \
        any(token.dep_ == "prep" or token.pos_ == "ADP" for token in doc):
            return True

        return False

    def validate(self, precondition):
        failed_checks = []
        
        # Check if the precondition is a valid string
        if not isinstance(precondition, str) or not precondition.strip():
            failed_checks.append("invalid_format")
            return {"valid": False, "failed_checks": failed_checks}
        
        """Validates a single precondition using POS tagging."""
        doc = self.nlp(precondition)
        
        # Rule 1: Starts with a noun or adjective (not an imperative verb) TODO difficult to find fitting rule for imperatives e.g.: Ensure that the map view is loaded.
        #if doc[0].pos_ in {"VERB"} and doc[0].tag_ in {"VB"}:  # "VB" is base form (imperative)
        #    failed_checks.append("avoids_commands")
        
        # Rule 2: Uses present tense (avoid future & past) but allow imperatives (with doc[0].tag_ != "VB")
        #if not any(token.tag_ in {"VBZ","VBP"} for token in doc) and doc[0].tag_ != "VB": 
        if not self.is_present_tense(doc):
            failed_checks.append("uses_present_tense")            
            #print(precondition)
            #else: #not verbs are present -> ok
        
        # Rule 3: Describes a state or requirement (e.g., contains "is", "must", "requires")
        if not self.describes_state_or_requirement(doc):
            failed_checks.append("describes_state_or_requirement")
            # TODO fails for "A valid list of waypoints with correct data" which is actually ok
            #               "Access to the system's coordinate specification module"


        return {"valid": not failed_checks, "failed_checks": failed_checks}

    def validate_all(self, preconditions):
        """Checks all preconditions in a given list and provides statistics."""
        total = len(preconditions)
        valid_count = 0
        failed_rule_counts = defaultdict(int)

        for precondition in preconditions:
            result = self.validate(precondition)
            if result["valid"]:
                valid_count += 1
            else:
                for rule in result["failed_checks"]:
                    failed_rule_counts[rule] += 1

        return {
            "total_preconditions": total,
            "valid_count": valid_count,
            "invalid_count": total - valid_count,
            "failed_rule_counts": dict(failed_rule_counts)
        }
