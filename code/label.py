import numpy as np
import json
import os

path = '/Users/lucasgonzalez/work/telecom/PRIM/code/data/100/label/'

offset = 500
rows = []

for file in os.listdir(path):
    if file.split(".")[-1]=='json':

        with open(path+file) as js:
            data = json.load(js)
            for bb in data['regions']:
                row = np.empty(10)
                row[0] = data['asset']['timestamp'] * 25 - offset
                row[1] = bb['tags'][0]
                row[2:4] = list(bb['boundingBox'].values())[2:4]
                row[4] = list(bb['boundingBox'].values())[1]
                row[5] = list(bb['boundingBox'].values())[0]
                row[6:] = -1.0
                if(row[0]>=0):
                    rows.append(row)

rows = np.array(rows)
rows = rows[rows[:,0].argsort()]
print(rows)

np.save("/Users/lucasgonzalez/work/telecom/PRIM/code/data/100/gt/gt.npy", rows)


