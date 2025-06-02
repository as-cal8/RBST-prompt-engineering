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
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
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

    def remove_entries_by_ids(self, dataset, ids_to_remove):
        """
        Removes entries from the dataset that match the given test_case_ids.
        
        Args:
            dataset (list): The original dataset (list of dicts).
            ids_to_remove (list): List of test_case_id strings to remove.
            
        Returns:
            list: Filtered dataset with unwanted entries removed.
        """
        return [entry for entry in dataset if entry[self.required_fields.get("test_case_id")] not in ids_to_remove]

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
                invalid_entries_id.append(entry[self.required_fields.get("test_case_id")])
                for field in missing_fields:
                    missing_field_counts[field] += 1

        #self.invalid_ids = combine_and_deduplicate(invalid_entries_id, self.invalid_ids) 

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "invalid_entries": total_entries - valid_entries,
            "missing_field_counts": dict(missing_field_counts),
            "invalid_entries_id": invalid_entries_id
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
                    invalid_entries_id.append(entry[self.required_fields.get("test_case_id")])
        
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
                    invalid_entries_id.append(entry[self.required_fields.get("test_case_id")])
                    continue  # Skip invalid entries
                
                try:
                    test_case_index = entry[self.required_fields.get("test_case_id")]
                except ValueError:
                    mismatches.append({self.required_fields.get("test_case_id"): entry[self.required_fields.get("test_case_id")], "error": "Invalid testCaseID format"})
                    invalid_entries_id.append(entry[self.required_fields.get("test_case_id")])
                    continue

                # Check if testCaseID corresponds to a valid index in original_requirements
                if test_case_index in original_requirements['issueid'].values:
                    expected_requirement = original_requirements.loc[original_requirements['issueid'] == test_case_index, 'RequirementText'].values[0]
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
                        invalid_entries_id.append(entry[self.required_fields.get("test_case_id")])
                else:
                    mismatches.append({self.required_fields.get("test_case_id"): test_case_index, "error": self.required_fields.get("test_case_id") + " out of range"})
                    invalid_entries_id.append(entry[self.required_fields.get("test_case_id")])

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
                        invalid_entries_id.append(entry[self.required_fields.get("test_case_id")])
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
                            invalid_entries_id.append(entry[self.required_fields.get("test_case_id")])
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
                    for test_step in entry[field_name]:
                        result = validator.validate(test_step)
                        
                        if result["valid"]:
                            valid_count += 1
                        else:
                            for rule in result["failed_checks"]:
                                failed_rule_counts[rule] += 1
                            invalid_entries_id.append(entry[self.required_fields.get("test_case_id")])
                else: # field empty
                    invalid_entries_id.append(entry[self.required_fields.get("test_case_id")])
                    
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
                        invalid_entries_id.append(entry[self.required_fields.get("test_case_id")])
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
        print("\nID check:")
        result_check_unique_field = self.check_unique_field(json_data) # invalid_entries_id
        print(result_check_unique_field)
        
        # verify testcase ID matches with the requirement from the req_reference
        print("\nRequirements similarity to reference:")
        result_validate_req = self.validate_requirements_with_similarity(json_data, req_reference)
        print(result_validate_req)
        
        # verify that test_objective entries are valid through rules
        print("\nTest objective:")
        result_val_test_objective = self.validate_test_objective(json_data)
        print(result_val_test_objective)
        
        # verify that preconditions entries are valid through rules
        print("\nPreconditions:")
        result_validate_preconditions = self.validate_preconditions(json_data)
        print(result_validate_preconditions)
        
        # verify that test step entries are valid through rules
        print("\nTest Steps:")
        result_validate_test_steps = self.validate_test_steps(json_data)
        print(result_validate_test_steps)
        
        # verify that expected result entry is valid
        print("\nExpected Result:")
        result_expected_result = self.validate_expected_result(json_data)
        print(result_expected_result)
        
        return self.invalid_ids




DATA_PATH = "C:/Users/alexs/Documents/Studium/Informatik/Seminar/RBST-prompt-engineering/datasets/dronology/dronology_with_id.csv"

# Example usage
if __name__ == "__main__":
    
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    
    json_names = [
        "results_chat_prompt0_zero_1_5b.json",
        "results_chat_prompt0_zero_8b.json",
        "results_chat_prompt0_zero_14b.json",
        "results_chat_prompt0_zero7b.json",
        "results_chat_prompt1_persona7b.json",
        "results_chat_prompt1_persona8b.json",
        "results_chat_prompt1_persona14b.json",
        "results_chat_prompt2_tot1_5b.json",
        "results_chat_prompt2_tot7b.json",
        "results_chat_prompt2_tot8b.json",
        "results_chat_prompt2_tot14b.json",
        "results_chat_prompt3_context_and_tot7b.json",
        "results_chat_prompt3_context_and_tot8b.json",
        "results_chat_prompt3_context_and_tot14b.json",
        "results_chat_prompt4_preparsing_and_tot_reqfirst7b.json",
        "results_chat_prompt4_preparsing_and_tot_reqfirst8b.json",
        "results_chat_prompt4_preparsing_and_tot_reqfirst14b.json",
    ]
    
    json_names_plotting = [
        "zero 1.5b",
        "zero 8b",
        "zero 14b",
        "zero 7b",
        "persona 7b",
        "persona 8b",
        "persona 14b",
        "tot 1.5b",
        "tot 7b",
        "tot 8b",
        "tot 14b",
        "cntxt tot 7b",
        "cntxt tot 8b",
        "cntxt tot 14b",
        "preparsing tot 7b",
        "preparsing tot 8b",
        "preparsing tot 14b",
    ]
    
    summary_data_invalid = []
    list_data_invalid = []
    summary_data_fieldcheck_fails = []
    list_data_fieldcheck  = []
    
    for json_name, json_name_plt in zip(json_names, json_names_plotting):
        print("\n\n")
        print("-> Evaluating: " + json_name)
        print("\n")

        # Define the file path relative to the script's directory
        json_path = script_dir / ("results/" + json_name) 

        df = pd.read_csv(DATA_PATH)
        requirements = df[['issueid', 'RequirementText']]
        
        comparer = TestCaseComparator()
        
        ds_zero = comparer.load_json(json_path)
        
        print("\nJson template check:")
        result_valid_entries = comparer.validate_json_entries(ds_zero)
        print(result_valid_entries)
        id_invalid_entries = result_valid_entries['invalid_entries_id']
        
        ds_zero_without_invalid_entries = comparer.remove_entries_by_ids(ds_zero, id_invalid_entries)
        requirements_without_invalid_entries = requirements[~df["issueid"].isin(id_invalid_entries)]
        
        id_field_check_fails = comparer.check_field_meaningfulness(ds_zero_without_invalid_entries, requirements_without_invalid_entries)
        
        print("\n")
        print(id_field_check_fails)
        #print(combine_and_deduplicate(id_invalid_entries, id_field_check_fails))
        print("\n\n")
        
        """
        Gather summary data for plots
        """
        summary_data_invalid.append((json_name_plt, result_valid_entries['total_entries'], result_valid_entries['valid_entries'], len(result_valid_entries['invalid_entries_id'])))
        summary_data_fieldcheck_fails.append((json_name_plt, 100, (len(ds_zero_without_invalid_entries) - len(id_field_check_fails))/len(ds_zero_without_invalid_entries), len(id_field_check_fails)/len(ds_zero_without_invalid_entries)))
        list_data_invalid.append((json_name_plt, result_valid_entries['invalid_entries_id']))
        list_data_fieldcheck.append((json_name_plt, [test['testCaseID'] for test in ds_zero_without_invalid_entries]))
    
    
    # Create DataFrame
    df_summary_fieldchek_fails = pd.DataFrame(summary_data_fieldcheck_fails, columns=["Model", "Total", "Valid", "Invalid"])
    df_summary_invalid = pd.DataFrame(summary_data_invalid, columns=["Model", "Total", "Valid", "Invalid"])



    # Set index and select columns
    df_plot = df_summary_fieldchek_fails.set_index("Model")[["Valid", "Invalid"]]
    df_plot = df_plot.sort_values(by=["Valid", "Invalid"], ascending=True)
    template_model_order = df_plot.index  # Save this ordering

    ax = df_plot.plot(kind='bar', stacked=True, figsize=(14, 4), colormap='tab20c', width=0.6)
    plt.title("Valid vs Invalid Entries per Model (Field checks)")
    plt.ylabel("Percent of Entries")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(script_dir / "results/visualization/comparison_chart_fieldcheck_invalid.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plotting: Stacked bar chart
    # Set index and select columns
    df_plot_json = df_summary_invalid.set_index("Model")[["Valid", "Invalid"]]
    df_plot_json = df_plot_json.loc[template_model_order]
    ax = df_plot_json.plot(kind='bar', stacked=True, figsize=(14, 4), colormap='tab20c', width=0.6)
    plt.title("Valid vs Invalid Entries per Model (JSON Template Check)")
    plt.ylabel("Number of Entries")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.ylim(0, 97)  # Set Y-axis limit
    plt.tight_layout()
    plt.savefig(script_dir / "results/visualization/comparison_chart_invalid.png", dpi=300, bbox_inches='tight')
    plt.show()


    # Add model size for grouping (extract from model name)
    def extract_size(model_name):
        for size in ['1.5b', '7b', '8b', '14b']:
            if size in model_name:
                return size
        raise NameError("???")

    # Helper to extract prompt type
    def extract_prompt_type(model_name):
        for prompt in ['zero', 'persona', 'tot', 'cntxt tot', 'preparsing tot']:
            if model_name.startswith(prompt):
                return prompt
        raise NameError(f"Unknown prompt type in: {model_name}")
    
    df_summary_invalid['Size'] = df_summary_invalid['Model'].apply(extract_size)
    df_summary_invalid['PromptType'] = df_summary_invalid['Model'].apply(extract_prompt_type)
    
    df_summary_fieldchek_fails['Size'] = df_summary_fieldchek_fails['Model'].apply(extract_size)
    df_summary_fieldchek_fails['PromptType'] = df_summary_fieldchek_fails['Model'].apply(extract_prompt_type)

    prompt_order = ['zero', 'persona', 'tot', 'cntxt tot', 'preparsing tot']
    size_order = ['1.5b', '7b', '8b', '14b']


    df_summary_invalid['PromptType'] = pd.Categorical(df_summary_invalid['PromptType'], categories=prompt_order, ordered=True)
    df_summary_invalid['Size'] = pd.Categorical(df_summary_invalid['Size'], categories=size_order, ordered=True)

    df_summary_fieldchek_fails['PromptType'] = pd.Categorical(df_summary_fieldchek_fails['PromptType'], categories=prompt_order, ordered=True)
    df_summary_fieldchek_fails['Size'] = pd.Categorical(df_summary_fieldchek_fails['Size'], categories=size_order, ordered=True)


    # --- PLOT 1: Stacked bar chart (Valid vs Invalid by size) ---
    df_counts_sorted = df_summary_invalid.sort_values(by=['Size', 'PromptType'])
    df_plot1 = df_counts_sorted.set_index('Model')[['Valid', 'Invalid']]
    df_plot1.plot(kind='bar', stacked=True, colormap='tab20c', figsize=(14, 6), width=0.6)
    plt.title('Valid vs Invalid Entries per Model (by Size)')
    plt.ylabel('Number of Entries')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.savefig(script_dir / "results/visualization/comparison_chart_invalid_models.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    # --- PLOT 2: Percentage bar plot (Valid %) grouped by Size ---
    df2_sorted = df_summary_fieldchek_fails.sort_values(by=['Size', 'PromptType'])
    df_plot2 = df2_sorted.set_index('Model')[['Valid', 'Invalid']]
    df_plot2.plot(kind='bar', stacked=True, colormap='tab20c', figsize=(14, 6), width=0.6)
    plt.title('Valid Percentage per Model (Grouped by Size)')
    plt.ylabel('Valid %')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(script_dir / "results/visualization/comparison_chart_fieldcheck_invalid_model.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    
    # Map model sizes to colors
    size_colors = {
        '1.5b': '#1f77b4',  # blue
        '7b': '#ff7f0e',    # orange
        '8b': '#2ca02c',    # green
        '14b': '#d62728',   # red
    }

    # Ensure 'Size' column exists
    df_summary_invalid['Size'] = df_summary_invalid['Model'].apply(extract_size)

    # Sort by Size, then Model name (prompt type implicitly sorted)
    df_sorted = df_summary_invalid.sort_values(by=['Size', 'Model'])

    # Plot manually with grouped bar colors
    fig, ax = plt.subplots(figsize=(14, 6))

    bottoms = [0] * len(df_sorted)
    x = range(len(df_sorted))

    # Bar for Valid
    bars_valid = ax.bar(
        x,
        df_sorted['Valid'],
        color=[size_colors[size] for size in df_sorted['Size']],
        label='Valid'
    )

    # Bar for Invalid stacked on top
    bars_invalid = ax.bar(
        x,
        df_sorted['Invalid'],
        bottom=df_sorted['Valid'],
        color=[size_colors[size] for size in df_sorted['Size']],
        alpha=0.4,
        label='Invalid'
    )

    # X-axis setup
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted['Model'], rotation=45, ha='right')
    ax.set_ylabel('Number of Entries')
    ax.set_title('Valid vs Invalid Entries per Model (by Size)')
    ax.grid(axis='y')

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=color, label=size) for size, color in size_colors.items()
    ]
    ax.legend(handles=legend_patches + [
        Patch(facecolor='gray', alpha=1.0, label='Valid'),
        Patch(facecolor='gray', alpha=0.4, label='Invalid')
    ])

    plt.tight_layout()
    plt.savefig(script_dir / "results/visualization/comparison_chart_invalid_models_colored.png", dpi=300, bbox_inches='tight')
    plt.show()