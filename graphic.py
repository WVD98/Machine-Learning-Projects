import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv(r'C:\Users\mathe\Desktop\Machine_Learning_Projects\Machine-Learning-Projects\air_traffic.xlsx',sep=';')

dataFrame = pd.DataFrame(data)



y = dataFrame['Pax']
x = dataFrame['Year']

plt.plot(x,y)

plt.xlabel("Datas") 
plt.ylabel("Número de voos Nacionais + Internacionais") 
plt.xticks(['jan/03', 'ago/08', 'dez/19', 'set/23', 'jul/03', 'fev/04', 'fev/08', 'jul/15', 'fev/15','abr/20', 'jul/20', 'jul/06', 'fev/06','jul/22', 'fev/23'], rotation = 90, fontsize = 13)
plt.xlim(0, 250) 
plt.ylim(0, 100000000) 
plt.locator_params(axis='both', nbins=30) 
plt.grid(color = "Red", axis='x')
plt.show()
# hello
