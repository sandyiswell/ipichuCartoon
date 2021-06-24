
import pandas as pd
import numpy as np

mydf = pd.read_csv('20006_20184_H_V_fitlerScores.csv')
print(mydf.head(5).to_string())

def metrics(DF):
    k = len(DF)
    s1 = DF.Filter_Vertical_Score
    s2 = DF.Filter_Horizontal_Score
    mydf["rms"] = np.sqrt((s1**2 + s2**2)/2)
    avg = []
    maximum = []
    minimum = []
    for i in range(k):
        average = np.mean([s1[i], s2[i]])
        avg.append(average)

        mx = np.max([s1[i], s2[i]])
        maximum.append(mx)

        mn = np.min([s1[i], s2[i]])
        minimum.append(mn)
    DF["avg"] = avg
    DF["max"] = maximum
    DF["min"] = minimum
    print(DF.to_string())
    DF.to_csv("20006_20184_H_V_filterScores_metrics.csv", index = False)
    return DF

metrics(mydf)



