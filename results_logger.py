import csv

def log_results(file_path, headers, values):
    file_exists = False
    try:
        with open(file_path, 'r') as f:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(values)
