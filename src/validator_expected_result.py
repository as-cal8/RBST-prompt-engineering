

class ExpectedResultValidator:
    def __init__(self):
        """Define general validation rules for expected result."""
        self.failed_checks = []

    def is_not_empty(self, expected_result):
        """Check if the field is not empty."""
        if not expected_result:
            self.failed_checks.append("is_not_empty")
            return False
        return True

    def is_string(self, expected_result):
        """Check if the expected result is a string."""
        if not isinstance(expected_result, str):
            self.failed_checks.append("is_string")
            return False
        return True

    def validate(self, expected_result):
        """Validates the given expected result based on defined rules."""
        
        # Reset the failed checks before validation
        self.failed_checks.clear()

        # Step 1: Check if expected_result is not empty
        self.is_not_empty(expected_result)
        
        # Step 2: Check if expected_result is a string
        self.is_string(expected_result)

        # If no failed checks, return valid, otherwise return failed checks
        return {"valid": not self.failed_checks, "failed_checks": self.failed_checks}
