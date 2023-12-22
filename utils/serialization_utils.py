import dill

def save_obj(obj, path):
    with open(path + '.pkl', 'wb') as f:
        dill.dump(obj, f)
        
def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return dill.load(f)
    