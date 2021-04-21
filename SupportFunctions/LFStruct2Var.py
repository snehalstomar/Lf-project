def LFStruct2Var(dict_in, keys):
    # keys must be a list or tuple containing the keys whose corresponding values
    # you want to access. For eg
    # a = {"Hello" : 1, "World" : 2, "Python" : 3}
    # p, q = LFStruct2Var(a, ("Hello", "World")) gives p = 1 and q = 2
    output = []
    for key in keys:
        output.append(dict_in[key])
    return output