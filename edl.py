import json
import numpy as np  # linear algebra
import math

from tqdm import tqdm

__all__ = ["extract"]


def extract(data_src):
    responses = []
    step = 200
    with open(data_src) as f:
        for line in tqdm(f, "reading data from %s" % data_src):
            data = json.loads(line)
            for i in range(0, len(data), step):
                if len(data[i: i + step]) < 2:
                    continue
                responses.append(data[i: i + step])

    return responses


def load_data(path, separate_char):
    f_data = open(path, 'r')
    data = []
    for lineID, line in enumerate(f_data):
        line = line.strip()
        # lineID starts from 0,按行处理内容
        if lineID % 3 == 0:
            temp = []
            student_id = int(lineID / 3)
            temp.append(student_id)
        # Q存储问题序列
        if lineID % 3 == 1:
            Q = line.split(separate_char)
            Q = list(map(int, Q))
            temp.append(Q)

        # A存储反应序列
        if lineID % 3 == 2:
            A = line.split(separate_char)
            A = list(map(int, A))
            temp.append(A)
            data.append(temp)

    f_data.close()
    return data
