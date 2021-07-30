import json
import csv

def to_csv():
    with open("./patientdata.json", "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    data_file = open('converted_csv.csv', 'w')
    csv_writer = csv.writer(data_file)

    # Counter variable used for writing  
    # headers to the CSV file 
    count = 0
    
    for emp in data: 
        if count == 0: 
    
            # Writing headers of CSV file 
            header = emp.keys() 
            csv_writer.writerow(header) 
            count += 1
    
        # Writing data of CSV file 
        csv_writer.writerow(emp.values())