from numpy import load

data = load('commands_mfcc_sets.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])