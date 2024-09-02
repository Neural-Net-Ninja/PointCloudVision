
file_path = r'K:\Navis Trees Labeling\Essen Reference Data\Essen3_p3.txt'


sem_class_ids = set()


with open(file_path, 'r') as file:

    header = next(file).strip().split(',')
    

    sem_class_id_index = header.index('SemClassID')
    
    # Process each line
    for line in file:

        columns = line.strip().split(',')
        
        sem_class_id = columns[sem_class_id_index]
        
        sem_class_ids.add(sem_class_id)

print(sem_class_ids)