# replace MATLAB struct with Python dictionary

def LFDefaultField(dictionary, key, value):
    try:
        dictionary[key] = value
    except:
        dictionary = {}
        dictionary[key] = value
    return dictionary