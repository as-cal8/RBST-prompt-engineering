import spacy
from collections import defaultdict

class TestObjectiveValidator:
    def __init__(self):
        """Load the spaCy NLP model and define validation rules."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Model not found. Try running: python -m spacy download en_core_web_sm")
        
    def validate(self, test_objective):
        """Checks if a test objective follows linguistic structure and validation rules."""
        doc = self.nlp(test_objective)
        failed_checks = []

        # Rule 1: Describes a goal → Starts with a verb (VB, VBP, VBZ)
        for token in doc: # Skip introductory words like "To"
            if token.pos_ not in {"PART", "DET"}:  # Ignore particles like "To"
                if token.pos_ != "VERB":
                    failed_checks.append("describes_goal")
                break

        # Rule 2: Uses active voice → Subject (nsubj) before the main verb TODO subjects often not recognized because unknown word / not part of dictionary
        #if any(token.dep_ == "nsubjpass" for token in doc):  # Passive subjects
        #    failed_checks.append("uses_active_voice")

        # Rule 3: Is concise → Word count <= 30
        if len(doc) > 40:
            failed_checks.append("is_concise")

        return {"valid": not failed_checks, "failed_checks": failed_checks}

    def validate_all(self, test_objectives):
        """Validates multiple test objectives and provides overall statistics."""
        total = len(test_objectives)
        valid_count = 0
        failed_rule_counts = defaultdict(int)

        for obj in test_objectives:
            result = self.validate(obj)
            if result["valid"]:
                valid_count += 1
            else:
                for rule in result["failed_checks"]:
                    failed_rule_counts[rule] += 1

        return {
            "total_test_objectives": total,
            "valid_count": valid_count,
            "invalid_count": total - valid_count,
            "failed_rule_counts": dict(failed_rule_counts)
        }
