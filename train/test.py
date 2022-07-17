import numpy as np

array = [[1,2,3,4,5],
         [1,2,3,4,5],
         [1,2,3,4,5],
         [1,2,3,4,5],
         [1,2,3,4,5]]

nparaay = np.asarray(array)
shapes = nparaay.shape
w = shapes[0]
h = shapes[1]

number = shapes[0]*shapes[1]
xi = 0
yi = 0
for i in range(number):
    print(nparaay[xi,yi])
    # xi = xi+1
    if yi != w-1:
        yi = yi+1
    else:
        xi = xi+1


#     num = i+1
#     if num % 2==0:
#         outlist = []
#         templist = list(nparaay[i])
#         lens = len(templist)
#
#         for k in range(len(templist)):
#             outlist.append(templist[lens-k-1])
#         print(outlist)
#     else:
#
#         print(nparaay[i])


