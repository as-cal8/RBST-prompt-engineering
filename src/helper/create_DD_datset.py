import json
import csv

# Load JSON data
with open('C:/Users/alexs/Documents/Studium/Informatik/Seminar/RBST-prompt-engineering/datasets/dronology/dronologydataset01.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare CSV output
with open('dronology_dd.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['RequirementText', 'issueid']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # Process each entry
    for entry in data['entries']:
        if 'DD' in entry['issueid'] and 'description' in entry['attributes']:
            writer.writerow({
                'RequirementText': entry['attributes']['description'],
                'issueid': entry['issueid']
            })

print("CSV file created successfully: dronology_dd.csv")