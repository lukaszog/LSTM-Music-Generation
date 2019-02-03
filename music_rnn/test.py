import cPickle 

with open("data/nottingham.pickle", 'rb') as f:
    pickle = cPickle.load(f)

print pickle[0:100]
