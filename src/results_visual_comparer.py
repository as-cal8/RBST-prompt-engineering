import json
from pathlib import Path
from tabulate import tabulate

# Configuration
RESULTS_DIR = "results"
PROMPT_FILES = [
    "results_chat_prompt0_zero.json",
    "results_chat_prompt1_persona.json",
    "results_chat_prompt2_tot.json",
    "results_chat_prompt3_context_and_tot.json",
    "results_chat_prompt4_preparsing_and_tot_reqfirst.json"
]
COMPARE_FIELDS = ["Requirement", "testObjective", "preconditions", "testSteps", "expectedResult"]

def load_results():
    """Load all result files into a dictionary"""
    results = {}
    for file in PROMPT_FILES:
        path = Path(RESULTS_DIR) / file
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                prompt_name = Path(file).stem.replace('results_chat_', '')
                results[prompt_name] = {item['testCaseID']: item for item in data}
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    return results

def display_field_comparison(test_case_id, results, field_index=0):
    """Display comparison for one field at a time"""
    field = COMPARE_FIELDS[field_index]
    comparison = []
    
    for prompt_name, prompt_data in results.items():
        if test_case_id in prompt_data:
            item = prompt_data[test_case_id]
            row = {'Prompt': prompt_name}
            
            # Get field value (handle nested fields)
            value = item
            for part in field.split('.'):
                value = value.get(part, 'N/A')
                if value == 'N/A':
                    break
            
            # Convert lists to strings with newlines
            if isinstance(value, list):
                value = '\n'.join(str(x) for x in value)
            row[field] = str(value)
            
            comparison.append(row)
    
    print(f"COMPARISON FOR TEST CASE: {test_case_id}")
    print(f"FIELD: {field}\n")
    print(tabulate(
        comparison,
        headers="keys",
        tablefmt="grid",
        maxcolwidths=[None, None],  # No column width limit
        stralign="left"
    ))
    print("\nPress Enter to see next field (or 'q' to quit)...")

def main():
    all_results = load_results()
    test_case_ids = set().union(*[set(p.keys()) for p in all_results.values()])
    
    print(f"Found {len(test_case_ids)} test cases")
    print("Sample IDs:", ", ".join(list(test_case_ids)[:3]))
    
    while True:
        test_case_id = input("\nEnter testCaseID to compare (or 'q' to quit): ").strip()
        if test_case_id.lower() == 'q':
            break
        
        if test_case_id not in test_case_ids:
            print(f"Error: {test_case_id} not found")
            continue
        
        # Cycle through each field with Enter
        field_index = 0
        while field_index < len(COMPARE_FIELDS):
            display_field_comparison(test_case_id, all_results, field_index)
            
            user_input = input().strip().lower()
            if user_input == 'q':
                break
            
            field_index += 1

if __name__ == "__main__":
    main()