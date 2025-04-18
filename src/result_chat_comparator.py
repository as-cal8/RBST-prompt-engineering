'''
    Loads two JSON datasets containing the chat of one of the prompt versions and 
    does qualitative comparison

Evaluation Criteria TODO

1. Ensure the generated JSON contains all required fields (e.g., Requirement, testObjective, preconditions, testSteps, expectedResult).
        Precision (Are all fields included?), Coverage (Are all fields filled meaningfully?)
        
2. The generated test descriptions must align with the specific requirement provided, ensuring that the details in the test case are consistent with the expected behavior outlined in the requirement.
        Metrics: Cosine similarity or contextual similarity between the original requirement and the generated testObjective, testSteps, etc.
        
        Similarity between datasets?
        
3. Assess whether the test descriptions are easy to understand for the intended users (e.g., developers, testers).
        Metrics: Readability scores such as Flesch-Kincaid Grade Level, syntactic simplicity, or manual reviews by human evaluators.

    Manual:
        Assess how technically accurate and aligned with domain-specific language (UAS, UAV systems, etc.) the generated content is.
        Metrics: Manual inspection for domain relevance or automatic keyword extraction and matching.
'''


import json
import os
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import pandas as pd
from pathlib import Path
from validator_test_objective import TestObjectiveValidator
from validator_precondition import PreconditionValidator
from validator_test_steps import TestStepsValidator
from validator_expected_result import ExpectedResultValidator

def combine_and_deduplicate(*lists):
    combined = set()
    for lst in lists:
        combined.update(lst)
    return list(combined)

class TestCaseComparator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the NLP model for contextual similarity.
        Uses a pre-trained Sentence Transformer model.
        """ 
        self.model = SentenceTransformer(model_name)
        
        # dict of required json fields which map to the actual string in the .json datasets generated
        self.required_fields = {
            "requirement": "Requirement",       # must match one entry of the original dataset
            "test_case_id": "testCaseID",       # must be unique and shall also be the index of the original requirement dataset
            "test_objective": "testObjective",  # rules, see TestObjectiveValidator
            "preconditions": "preconditions",   # 
            "test_steps": "testSteps",          # 
            "expected_result": "expectedResult" # 
        }
        
        self.invalid_ids = []

    def load_json(self, filename):
        """Loads a JSON dataset from a file."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)

    def validate_json_entries(self, json_data):
        """
        Validates JSON entries against required fields and returns statistics.

        Args:
            json_data (list): List of dictionaries representing JSON entries.
            required_fields (list): List of required field names.

        Returns:
            dict: Statistics on valid and invalid entries.
        """
        total_entries = len(json_data)
        valid_entries = 0
        missing_field_counts = defaultdict(int)
        invalid_entries_id = []

        for id, entry in enumerate(json_data):
            missing_fields = [field for field in self.required_fields.values() if field not in entry or entry[field] in [None, ""]]            
            
            if not missing_fields:
                valid_entries += 1
            else:
                invalid_entries_id.append(id)
                for field in missing_fields:
                    missing_field_counts[field] += 1

        self.invalid_ids = combine_and_deduplicate(invalid_entries_id, self.invalid_ids) 

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "invalid_entries": total_entries - valid_entries,
            "missing_field_counts": dict(missing_field_counts),
            "invalid_entries_id": self.invalid_ids
        }

    def check_unique_field(self, json_data):
        field_name = self.required_fields.get("test_case_id")
        """
        Checks if a specific testCaseID is unique across all entries in a dataset.

        Args:
            json_data (list): List of dictionaries representing JSON entries.
            field_name (str): The field to check for uniqueness.

        Returns:
            dict: Summary of unique and duplicate values.
        """
        
        value_counts = defaultdict(int)
        invalid_entries_id = []
        
        # Count occurrences of each field value
        for id, entry in enumerate(json_data):
            if id not in self.invalid_ids:    
                if field_name in entry and entry[field_name] not in [None, ""]:
                    value_counts[entry[field_name]] += 1
                else:
                    invalid_entries_id.append(id)
        
        # Find duplicates
        duplicates = {key: count for key, count in value_counts.items() if count > 1}

        self.invalid_ids = combine_and_deduplicate(invalid_entries_id, self.invalid_ids) 

        return {
            "total_entries": len(json_data),
            "unique_values": len(value_counts) - len(duplicates),
            "duplicate_values": len(duplicates),
            "duplicates": duplicates,  # Dictionary showing which values are duplicated and how many times
            "invalid_entries_id": invalid_entries_id
        }
    
    def validate_requirements_with_similarity(self, json_data, original_requirements, threshold=0.95):
        """
        Checks if the 'Requirement' field in each JSON entry is similar to the corresponding 
        requirement from the original list using cosine similarity.

        Args:
            json_data (list): List of JSON test case dictionaries.
            original_requirements (list): List of original requirement strings.
            threshold (float): Minimum similarity score (0 to 1) for a match.

        Returns:
            dict: Summary of matching and mismatching entries based on similarity.
        """
        total_entries = len(json_data)
        mismatches = []
        invalid_entries_id = []

        for id, entry in enumerate(json_data):
            if id not in self.invalid_ids:
                if self.required_fields.get("test_case_id") not in entry or self.required_fields.get("requirement") not in entry:
                    invalid_entries_id.append(id)
                    continue  # Skip invalid entries
                
                try:
                    test_case_index = int(entry[self.required_fields.get("test_case_id")])  # Ensure testCaseID is an integer
                except ValueError:
                    mismatches.append({self.required_fields.get("test_case_id"): entry[self.required_fields.get("test_case_id")], "error": "Invalid testCaseID format"})
                    invalid_entries_id.append(id)
                    continue

                # Check if testCaseID corresponds to a valid index in original_requirements
                if 0 <= test_case_index < len(original_requirements):
                    expected_requirement = original_requirements[test_case_index]
                    actual_requirement = entry[self.required_fields.get("requirement")]

                    # Compute similarity score
                    embeddings = self.model.encode([expected_requirement, actual_requirement], convert_to_tensor=True)
                    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

                    if similarity_score < threshold:
                        mismatches.append({
                            self.required_fields.get("test_case_id"): test_case_index,
                            "expected": expected_requirement,
                            "found": actual_requirement,
                            "similarity_score": round(similarity_score, 2)
                        })
                        invalid_entries_id.append(id)
                else:
                    mismatches.append({self.required_fields.get("test_case_id"): test_case_index, "error": self.required_fields.get("test_case_id") + " out of range"})
                    invalid_entries_id.append(id)

        self.invalid_ids = combine_and_deduplicate(invalid_entries_id, self.invalid_ids) 

        return {
            "total_entries": total_entries,
            "valid_entries": total_entries - len(mismatches),
            "mismatched_entries": len(mismatches),
            "details": mismatches,
            "invalid_entries_id": invalid_entries_id
        }

    def validate_test_objective(self, json_data):
        validator = TestObjectiveValidator()
        field_name = self.required_fields.get("test_objective")
        total_entries = len(json_data)
        
        # Counters for valid and invalid objectives
        valid_count = 0
        failed_rule_counts = defaultdict(int)
        invalid_entries_id = []

        for id, entry in enumerate(json_data):
            if id not in self.invalid_ids:
                if field_name in entry and entry[field_name] not in [None, ""]:
                    result = validator.validate(entry[field_name])

                    if result["valid"]:
                        valid_count += 1
                    else:
                        for rule in result["failed_checks"]:
                            failed_rule_counts[rule] += 1
                        invalid_entries_id.append(id)
                # elif field_name in entry and entry[field_name] in [None, ""]:
                    #test_objective field empty 
                # else:
                    # ?

        self.invalid_ids = combine_and_deduplicate(invalid_entries_id, self.invalid_ids) 

        return {
            "total_entries": total_entries,
            "valid_count": valid_count,
            "invalid_count": total_entries - valid_count,
            "failed_rule_counts": dict(failed_rule_counts),  # Convert defaultdict to normal dict for output
            "invalid_entries_id": invalid_entries_id
        }

    def validate_preconditions(self, json_data):
        validator = PreconditionValidator()
        field_name = self.required_fields.get("preconditions")
        total_entries = 0
        
        # Counters for valid and invalid objectives
        valid_count = 0
        failed_rule_counts = defaultdict(int)
        invalid_entries_id = []

        for id, entry in enumerate(json_data):
            if id not in self.invalid_ids:
                if field_name in entry and entry[field_name] not in [None, ""]:
                    total_entries += len(entry[field_name])
                    for precond in entry[field_name]:
                        result = validator.validate(precond)
                        
                        if result["valid"]:
                            valid_count += 1
                        else:
                            for rule in result["failed_checks"]:
                                failed_rule_counts[rule] += 1
                            invalid_entries_id.append(id)
                # elif field_name in entry and entry[field_name] in [None, ""]:
                    #test_objective field empty 
                # else:
                    # ?
        
        self.invalid_ids = combine_and_deduplicate(invalid_entries_id, self.invalid_ids) 
        
        return {
            "total_entries": total_entries,
            "valid_count": valid_count,
            "invalid_count": total_entries - valid_count,
            "failed_rule_counts": dict(failed_rule_counts),  # Convert defaultdict to normal dict for output
            "invalid_entries_id": invalid_entries_id
        }

    def validate_test_steps(self, json_data):
        validator = TestStepsValidator()
        field_name = self.required_fields.get("test_steps")
        total_entries = 0
        
        # Counters for valid and invalid objectives
        valid_count = 0
        failed_rule_counts = defaultdict(int)
        invalid_entries_id = []

        for id, entry in enumerate(json_data):
            if id not in self.invalid_ids:
                if field_name in entry and entry[field_name] not in [None, ""]:
                    total_entries += len(entry[field_name])
                    for test_steps in entry[field_name]:
                        result = validator.validate(test_steps)
                        
                        if result["valid"]:
                            valid_count += 1
                        else:
                            for rule in result["failed_checks"]:
                                failed_rule_counts[rule] += 1
                            invalid_entries_id.append(id)
                else: # field empty
                    invalid_entries_id.append(id)
                    
        self.invalid_ids = combine_and_deduplicate(invalid_entries_id, self.invalid_ids) 
        
        return {
            "total_entries": total_entries,
            "valid_count": valid_count,
            "invalid_count": total_entries - valid_count,
            "failed_rule_counts": dict(failed_rule_counts),  # Convert defaultdict to normal dict for output
            "invalid_entries_id": invalid_entries_id
        }

    def validate_expected_result(self, json_data):
        validator = ExpectedResultValidator()
        field_name = self.required_fields.get("expected_result")
        total_entries = len(json_data)
        
        # Counters for valid and invalid objectives
        valid_count = 0
        failed_rule_counts = defaultdict(int)
        invalid_entries_id = []

        for id, entry in enumerate(json_data):
            if id not in invalid_entries_id:
                if field_name in entry and entry[field_name] not in [None, ""]:
                    result = validator.validate(entry[field_name])

                    if result["valid"]:
                        valid_count += 1
                    else:
                        for rule in result["failed_checks"]:
                            failed_rule_counts[rule] += 1
                        invalid_entries_id.append(id)
                # elif field_name in entry and entry[field_name] in [None, ""]:
                    #test_objective field empty 
                # else:
                    # ?

        self.invalid_ids = combine_and_deduplicate(invalid_entries_id, self.invalid_ids) 

        return {
            "total_entries": total_entries,
            "valid_count": valid_count,
            "invalid_count": total_entries - valid_count,
            "failed_rule_counts": dict(failed_rule_counts),  # Convert defaultdict to normal dict for output
            "invalid_entries_id": invalid_entries_id
        }

    def check_field_meaningfulness(self, json_data, req_reference):
        # are added manually but still for safety purposes lets check if ID is unique
        result_check_unique_field = self.check_unique_field(json_data) # invalid_entries_id
        print(result_check_unique_field)
        
        # verify testcase ID matches with the requirement from the req_reference
        print("Requirements similarity to reference:")
        result_validate_req = self.validate_requirements_with_similarity(json_data, req_reference)
        print(result_validate_req)
        
        # verify that test_objective entries are valid through rules
        print("Test objective:")
        result_val_test_objective = self.validate_test_objective(json_data)
        print(result_val_test_objective)
        
        # verify that preconditions entries are valid through rules
        print("Preconditions:")
        result_validate_preconditions = self.validate_preconditions(json_data)
        print(result_validate_preconditions)
        
        # verify that test step entries are valid through rules
        print("Test Steps:")
        result_validate_test_steps = self.validate_test_steps(json_data)
        print(result_validate_test_steps)
        
        # verify that expected result entry is valid
        print("Expected Result:")
        result_expected_result = self.validate_expected_result(json_data)
        print(result_expected_result)
        
        return self.invalid_ids



DATA_PATH = "C:/Users/alexs/Documents/Studium/Informatik/Seminar/RBST-prompt-engineering/datasets/dronology/dronology_with_id.csv"

# Example usage
if __name__ == "__main__":
    
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    # Define the file path relative to the script's directory
    json_path = script_dir / "results_chat_prompt0_zero.json"          # [1, 4, 69, 90, 71, 74, 75, 11, 46, 18, 87, 56, 62, 58, 91, 28, 29, 94]
    #json_path = script_dir / "results_chat_prompt1_persona.json"       # [6, 7, 9, 10, 25, 29, 44, 45, 46, 47, 49, 55, 57, 58, 64, 69, 71, 72, 75, 76, 77, 78, 79, 82, 87, 91, 96]
    #json_path = script_dir / "results_chat_prompt2_tot.json"           # [9, 10, 11, 29, 42, 44, 46, 56, 57, 58, 61, 71, 78, 82, 87, 89, 90, 91, 95, 96]
    #json_path = script_dir / "results_chat_prompt3_context_simple.json"# [96, 33, 67, 58, 35, 3, 71, 40, 41, 74, 43, 44, 77, 46, 14, 48, 90, 91]
    #json_path = script_dir / "results_chat_prompt4_preparsing_and_tot.json"# 
    #json_path = script_dir / "results_chat_prompt4_preparsing_and_tot_reqfirst.json"# [71, 8, 44, 45, 46, 82, 88, 58, 91, 92, 29]
    
    df = pd.read_csv(DATA_PATH)
    requirements = df['RequirementText']
    
    comparer = TestCaseComparator()
    
    ds_zero = comparer.load_json(json_path)
    
    print("Json template check:")
    result_valid_entries = comparer.validate_json_entries(ds_zero)
    print(result_valid_entries)
    id_invalid_entries = result_valid_entries['invalid_entries_id']
    
    id_field_check_fails = comparer.check_field_meaningfulness(ds_zero, requirements)
    print(combine_and_deduplicate(id_invalid_entries, id_field_check_fails))
    
    
    #similarity_results = comparer.compare_datasets("dataset1.json", "dataset2.json")

    # Print results
    #for testCaseID, scores in similarity_results.items():
    #    print(f"\nTestCaseID: {testCaseID}")
    #    for field, score in scores.items():
    #        print(f"  {field} Similarity: {score:.2f}")
