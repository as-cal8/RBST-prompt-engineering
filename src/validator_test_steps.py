import re

class TestStepsValidator:
    def __init__(self):
        """Define general validation rules for test steps."""
        self.failed_checks = []

    def is_not_empty(self, test_steps):
        """Check if the field is not empty."""
        if not test_steps:
            self.failed_checks.append("is_not_empty")
            return False
        return True

    def is_list_or_string(self, test_steps):
        """Check if the input is either a list or a string."""
        if not isinstance(test_steps, (str, list)):
            self.failed_checks.append("is_list_or_string")
            return False
        return True

    def is_list_numbered(self, test_steps):
        """Check if the list is numbered properly (if it is a list)."""
        if isinstance(test_steps, list):
            if not all(isinstance(item, str) and re.match(r"^\d+\.\s", item) or isinstance(item, str) for item in test_steps):
                self.failed_checks.append("is_list_numbered")
                return False
        return True

    def validate(self, test_steps):
        """Validates the given test steps based on defined rules."""
        
        # Reset the failed checks before validation
        self.failed_checks.clear()

        # Step 1: Check if test_steps is not empty
        # self.is_not_empty(test_steps) # just ignore empty entries
        
        # Step 2: Check if test_steps is either a string or a list
        self.is_list_or_string(test_steps)
        
        # Step 3: If it's a list, check if it's numbered (optional)
        self.is_list_numbered(test_steps)

        # If no failed checks, return valid, otherwise return failed checks
        return {"valid": not self.failed_checks, "failed_checks": self.failed_checks}
