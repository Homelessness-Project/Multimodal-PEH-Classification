import csv
import argparse

def tsv_to_csv(tsv_file, csv_file):
    # Open the TSV file for reading and the CSV file for writing
    with open(tsv_file, 'r', newline='', encoding='utf-8') as tsv_in, \
         open(csv_file, 'w', newline='', encoding='utf-8') as csv_out:
        # Use csv.reader with delimiter='\t' to read the TSV file
        reader = csv.reader(tsv_in, delimiter='\t')
        writer = csv.writer(csv_out)
        
        # Write each row from the TSV file to the CSV file
        for row in reader:
            writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a TSV file to a CSV file.')
    parser.add_argument('tsv_file', help='Path to the input TSV file')
    parser.add_argument('csv_file', help='Path to the output CSV file')
    args = parser.parse_args()
    
    tsv_to_csv(args.tsv_file, args.csv_file)
