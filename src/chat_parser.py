'''
Tracks prompt + responses for single requirement prompts.

{
  "testCaseID": "TC-001",
  "testObjective": "Verify that the login functionality allows users to log in with valid credentials.",
  "preconditions": [
    "User must be registered in the system.",
    "The application must be running."
  ],
  "testSteps": [
    "1. Open the login page.",
    "2. Enter a valid username and password.",
    "3. Click the 'Login' button."
  ],
  "expectedResult": "User is successfully logged in and redirected to the dashboard."
}
'''

import json
import os

class TestCaseChatParser:
    def __init__(self):
        """Initialize an empty list to store test cases."""
        self.data = []

    def _parse_and_validate(self, json_string):
        """Parses, validates, and removes unnecessary whitespaces."""
        try:
            if not json_string.strip():
                return {}  # Allow empty JSON

            entry = json.loads(json_string)

            # Trim unnecessary whitespaces from all string fields
            for key, value in entry.items():
                if isinstance(value, str):
                    entry[key] = value.strip()
                elif isinstance(value, list):
                    entry[key] = [item.strip() if isinstance(item, str) else item for item in value]

            return entry
        except (json.JSONDecodeError, ValueError) as e:
            #raise ValueError(f"Invalid JSON input: {e}")
            return {}  # Allow empty JSON

    def count_empty_entries(self):
        """Counts the number of test cases with an empty JSON structure."""
        return sum(1 for entry in self.data if not entry or not any(k for k in entry if k not in {"testCaseID", "prompt"}))

    def print_empty_entry_ids(self):
        """Prints all testCaseIDs where JSON content is empty."""
        empty_ids = [entry['testCaseID'] for entry in self.data if not entry or not any(k for k in entry if k not in {"testCaseID", "prompt"})]
        if empty_ids:
            print("Test Case IDs with empty JSON content:", empty_ids)
        else:
            print("No test cases with empty JSON content.")

    def add_entry(self, testCaseID, json_string, prompt_strings):
        """
        Adds a new test case entry after parsing and validating it.
        Also adds a list of prompt strings to the entry.
        Checks if testCaseID is unique.
        """
        # Check if the testCaseID is unique
        if any(entry['testCaseID'] == testCaseID for entry in self.data):
            raise ValueError(f"testCaseID '{testCaseID}' already exists. Please use a unique ID.")

        # Parse and validate the JSON string
        entry = self._parse_and_validate(json_string)
        
        # Add the testCaseID and prompt strings to the entry
        entry['testCaseID'] = testCaseID
        
        # Validate and add the prompts
        if isinstance(prompt_strings, list):
            entry['prompt'] = [
                prompt.get('content', '').strip()  # Extract 'content' and strip whitespace
                for prompt in prompt_strings
                if isinstance(prompt, dict) and 'content' in prompt  # Ensure it's a dict with 'content'
            ]
        else:
            entry['prompt'] = [prompt_strings.strip()]  # Handle single string case

        self.data.append(entry)
        
    def save(self, filename):
        """Saves the current test cases to a JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as file:
                json.dump(self.data, file, indent=4)
            print(f"Data successfully saved to {filename}")
        except IOError as e:
            print(f"Error saving file: {e}")

    def load(self, filename):
        """Loads test cases from a JSON file."""
        if not os.path.exists(filename):
            print(f"File {filename} not found.")
            return
        
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                self.data = json.load(file)

            # Validate structure after loading
            for entry in self.data:
                required_fields = {"testCaseID", "testObjective", "preconditions", "testSteps", "expectedResult"}
                if not all(field in entry for field in required_fields):
                    raise ValueError(f"Invalid test case structure in file: {filename}")

            print(f"Data successfully loaded from {filename}")
        except (json.JSONDecodeError, ValueError, IOError) as e:
            print(f"Error loading file: {e}")

    def get_json(self):
        """Returns the current test cases as a JSON string."""
        return json.dumps(self.data, indent=4)
