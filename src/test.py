import dill

path = 'C:\\Users\\gavin\Documents\\AMP\\admitted_matriculation_predictor\\test2.pkl'

with open(path, 'rb') as f:

    obj = dill.load(f)

print(obj.df)