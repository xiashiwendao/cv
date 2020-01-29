import os,sys
os.pardir
sys.call_tracing.__doc__
help(sys.call_tracing)
callable(sys.call_tracing)

help(os)
os.__doc__

dir(os)
dir(list)
import numpy
dir(numpy.array)
a = numpy.array([1,2,3,4,5])
a.size
dir(a)

def retArray():
    return 4

b = retArray()
dir(b)
c = retArray()
b.__len__
sorted(set(dir(b)) - set(dir(object)))[::-1]

class container():
    __name = ''
    __school = ''
    def setName(self, user_name):
        self.name = user_name
    
    def getName(self):
        return self.name

con = container()
con.setName('Lorry')
print(con.getName())
con.__dict__
container.__dict__
container.__na

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=8888)
model.__dict__

model2 = RandomForestClassifier(max_depth=9999)
model2.__dict__
