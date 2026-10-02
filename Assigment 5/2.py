import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic = pd.read_csv('C:\\Documentos\\2SEM2026\\ArtificialIntelligence\\Datasets\\titanic.csv')
sns.countplot(x='Survived', data=titanic, hue='Sex')
plt.xticks([0, 1], ['No', 'Sí'])
plt.xlabel('survived')
plt.ylabel('count')
plt.show() 

sns.countplot(x='Survived', data=titanic, hue='Pclass')
plt.xticks([0, 1], ['No', 'Sí'])
plt.xlabel('survived')
plt.ylabel('count')
plt.show()
