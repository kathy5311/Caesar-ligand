import numpy as np

f='test_subset_rev0718_2.npz'

data=np.load(f, allow_pickle=True)

#print(list(data.keys()))

#print(len(data['tags']))
#print(data['features'])

for i in list(data['features'][0].keys()):
    print(i)
    print(data['features'][-1][i])
    print(data['features'][-1][i].shape)
    print()