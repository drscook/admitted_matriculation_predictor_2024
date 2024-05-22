import dill

path = 'C:\\Users\\gavin\Documents\\AMP\\admitted_matriculation_predictor\\119\\X.pkl'

with open(path, 'rb') as f:

    obj = dill.load(f)

# print(obj.df)