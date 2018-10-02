
import glob
import numpy as np

output_file = "percentiles.csv"
f = open(output_file, "w+")
f.write("model;length;optim;batch;95\n")
f.close()

# List all csv files in src folder
csvFileNames = glob.glob("results/*.csv")


for csvFileName in csvFileNames:
    with open(csvFileName, newline='') as csvFile:
        contents = np.loadtxt(csvFile, delimiter=";", skiprows=1)
        accuracies = contents[:, 1]
        params = csvFileName.split("_")
        params[0] = params[0].replace("results/", "")
        params[1] = params[1].replace("len", "")
        params[3] = params[3].replace("batch", "")
        params[3] = params[3].replace(".csv", "")
        output = params
        output.append(str(np.percentile(accuracies, 95)))
        f = open(output_file, "a+")
        f.write("%s\n" % ";".join(output))
        f.close()