import numpy as np
lst = [i for i in range(300)]
print(lst)
arr = np.array(lst)
arr[64:-64]

ma = np.zeros((20, 20))
ma[5:-2, 7:-7] = 1
ma

t = (10,)

for item in t:
    print(item)

def testLocalvar(flag):
    if flag:
        v = 'true'
    else:
        v='false'
    
    ret = v + ' return'

    return ret

testLocalvar(True)

def bb():
    return external_var + "hello"

external_var = "ok, "
bb()

def bb2():
    def cc():
        return external_var
    
    ret = cc()
    return ret + "hello2"

external_var = "ok, "
bb2()

