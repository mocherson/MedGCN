import pickle as pk



def save_obj(obj, name ):
    with open(name , 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name , 'rb') as f:
        return pk.load(f)