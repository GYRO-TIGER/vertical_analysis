import pickle

file_path = 'input_subjects.pickle'  # Replace with the actual path to your .pickle file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

print(data)