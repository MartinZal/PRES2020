import pandas as pd
import matplotlib.pyplot as plt
import re

Tamb = pd.read_csv('data/17_07_2019_Tamb.txt', delimiter=';')

plt.figure(1)
plt.plot(Tamb.values[:, 2].tolist())
plt.show()

Tamb_val = Tamb.values[:, 2].tolist()
Tamb_val_clean = []
for i in range(len(Tamb_val)):
    Tamb_val_clean.append(re.sub('\+', '', Tamb_val[i]))
    Tamb_val_clean[i] = Tamb_val_clean[i].replace(',', '.')
    Tamb_val_clean[i] = float(Tamb_val_clean[i])



print(Tamb_val_clean)
print(type(Tamb_val_clean[10]))