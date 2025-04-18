import json
import csv
from difflib import SequenceMatcher

def similar(a, b):
    """Calculate text similarity ratio between two strings"""
    return SequenceMatcher(None, a, b).ratio()

def match_requirements(json_data, csv_rows):
    """Match JSON entries to CSV rows based on requirement text"""
    # Create a dictionary to store issueid matches
    matches = []
    
    for csv_row in csv_rows:
        csv_text = csv_row['RequirementText']
        best_match = None
        highest_similarity = 0
        
        for entry in json_data['entries']:
            # Skip non-requirement entries
            if entry['attributes']['issuetype'] != 'Requirement':
                continue
                
            json_text = entry['attributes']['description']
            similarity = similar(csv_text, json_text)
            
            if similarity > highest_similarity and similarity > 0.9: 
                highest_similarity = similarity
                best_match = entry['issueid']
        
        # Add the match to results
        matches.append({
            **csv_row,
            'issueid': best_match if best_match else 'NOT_FOUND'
        })
    
    return matches

def main():
    # Load JSON data
    try:
        with open("C:/Users/alexs/Documents/Studium/Informatik/Seminar/RBST-prompt-engineering/datasets/dronology/dronologydataset01.json", "r", encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # Load CSV data
    try:
        with open("C:/Users/alexs/Documents/Studium/Informatik/Seminar/RBST-prompt-engineering/datasets/dronology/dronology.csv", 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            csv_rows = list(csv_reader)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Match requirements
    matched_data = match_requirements(json_data, csv_rows)

    # Write updated CSV
    try:
        with open('dataset/dronology/dronology_with_id.csv.csv', 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['ProjectID', 'RequirementText', 'IsFunctional', 'IsQuality', 'issueid']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(matched_data)
        
        print("Successfully created updated_requirements.csv with issue IDs")
    except Exception as e:
        print(f"Error writing updated CSV: {e}")

if __name__ == '__main__':
    main()