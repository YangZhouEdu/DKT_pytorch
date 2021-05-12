import numpy as np
import math
import tqdm
import itertools
class DataReader():
    def __init__(self,path,separate_char):
        self.path = path
        self.separate_char = separate_char

        ### data format
        ### 15
        ### 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
        ### 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
    def load_data(self):
        data = []
        f_data = open(self.path, 'r')
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0,按行处理内容
            if lineID % 3 == 0:
                temp = []
                student_id = int(lineID / 3)
                temp.append(student_id)
            # Q存储问题序列
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)
                Q = list(map(int, Q))
                temp.append(Q)

            # A存储反应序列
            if lineID % 3 == 2:
                A = line.split(self.separate_char)
                A = list(map(int, A))
                temp.append(A)
                data.append(temp)

        f_data.close()
        return data