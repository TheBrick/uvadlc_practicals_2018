import glob
import numpy as np
import matplotlib.pyplot as plt


csvFileNames = glob.glob("results/*.csv")

for csvFileName in csvFileNames:
    with open(csvFileName, newline='') as csvFile:
        contents = np.loadtxt(csvFile, delimiter=";", skiprows=1)
        plt.plot(contents[:,0], contents[:,1])
        plt.ylim(0.0, 1.1)
        plt.title(csvFileName)
        plt.show()