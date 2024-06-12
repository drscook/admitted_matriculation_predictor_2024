import dill

path = 'C:\\Users\\Gavin\Documents\\admitted_matriculation_predictor\\98\\X.pkl'

with open(path, 'rb') as f:

    obj = dill.load(f)

print(obj)