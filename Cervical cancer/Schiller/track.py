import pickle
count = 1
ms = []
ns = []
scores = []
accuracies = []
sensitivities = []
specificities = []
positive_Predictive_Accuracies = []
negative_Predictive_Accuracies = []
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
fig,ax = plt.subplots(subplot_kw={'projection':'3d'})
score = 0
index = 0
while count < 400:
    output = open('meta'+str(count)+'.pkl', 'rb')
    data = pickle.load(output)
    output.close()
    ms.append(data['m'])
    ns.append(data['n'])
    scores.append(data['score'])
    accuracies.append(data['accuracy'])
    sensitivities.append(data['Sensitivity'])
    specificities.append(data['Specificity'])
    positive_Predictive_Accuracies.append(data['Positive_Predictive_Accuracy '])
    negative_Predictive_Accuracies.append(data['Negative_Predictive_Accuracy:'])
    if data['score'] > score:
        score = data['score']
        index = count
    if count%20 == 0:
        # ax.plot(ms,ns,accuracies)
        # ax.plot(ms,ns,sensitivities)
        # ax.plot(ms,ns,specificities)
        # ax.plot(ms,ns,positive_Predictive_Accuracies)
        # ax.plot(ms,ns,negative_Predictive_Accuracies)
        ax.plot(ms,ns,scores)
        ms = []
        ns = []
        scores = []
        accuracies = []
        sensitivities = []
        specificities = []
        positive_Predictive_Accuracies = []
        negative_Predictive_Accuracies = []
    count = count + 1
##################################################################################
ax.set_xlabel('positive clusters')
ax.set_ylabel('nagative clusters')
ax.set_zlabel('calssification score')
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
print ('score: ', score)
print ('index: ',index)
plt.show()