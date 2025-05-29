import re
from validator_common import CommonValidator

class TestStepsValidator(CommonValidator):
    def __init__(self):
        super().__init__()
        """Define general validation rules for test steps."""
        self.failed_checks = []

    def is_not_empty(self, test_step):
        """Check if the field is not empty."""
        if not test_step:
            self.failed_checks.append("is_not_empty")
            return False
        return True

    def is_list_or_string(self, test_step):
        """Check if the input is either a list or a string."""
        if not isinstance(test_step, (str, list)):
            self.failed_checks.append("is_list_or_string")
            return False
        return True

    def is_list_numbered(self, test_step):
        """Check if the list is numbered properly (if it is a list)."""
        if isinstance(test_step, list):
            if not all(isinstance(item, str) and re.match(r"^\d+\.\s", item) or isinstance(item, str) for item in test_step):
                self.failed_checks.append("is_list_numbered")
                return False
        return True

    def is_teststep_present_tense(self, test_step):
        
        if not isinstance(test_step, str):
            return False
        
        # Strip leading/trailing whitespace
        test_step = test_step.strip()
        
        if not test_step:
            return False  # Empty string not allowed
        
        numbered_pattern = r'^\d+\.(\s.+\.)$'
        match = re.match(numbered_pattern, test_step)
        if match:
            match= match.group(1).strip()
        else:
            sentence_pattern = r'^(\s.+\.)$'
            match = re.match(sentence_pattern, test_step)
            if match:
                match = match.group(1).strip()
        
        if match:
            """Validates a single precondition using POS tagging."""
            doc = self.nlp(match)
            
            return  self.is_present_tense(doc)
        else:
            return False

    def validate(self, test_step):
        """Validates the given test steps based on defined rules."""
        
        # Reset the failed checks before validation
        self.failed_checks.clear()

        # Step 1: Check if test_steps is not empty
        # self.is_not_empty(test_steps) # just ignore empty entries
        
        # Step 2: Check if test_steps is either a string or a list
        is_list = self.is_list_or_string(test_step)
        
        # Step 3: If it's a list, check if it's numbered (optional)
        is_numbered = self.is_list_numbered(test_step)
        
        if (not self.is_teststep_present_tense(test_step)):
            self.failed_checks.append("uses_present_tense")            

        # If no failed checks, return valid, otherwise return failed checks
        return {"valid": not self.failed_checks, "failed_checks": self.failed_checks}
