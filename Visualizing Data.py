# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 13:44:31 2023

@author: swank
"""

#Choosing the Right Graph¶
#Showing parts of a whole with pie charts
import matplotlib.pyplot as plt
%matplotlib inline
#  ^  This'll never run for my Spyder program
# Reminder    ...   [.ipynb will work]

#All plots run happily in Jupyter
# Runs --> Boxplots, Bargraphs, Scatterplots
​
values = [5, 8, 9, 10, 4, 7]
colors = ['b', 'g', 'r', 'c', 'm', 'y']
labels = ['A', 'B', 'C', 'D', 'E', 'F']
explode = (0, 0.2, 0, 0, 0, 0)

# some arrays
​
plt.pie(values, colors=colors, labels=labels, 
        explode=explode, autopct='%1.1f%%',
        counterclock=False, shadow=True)
plt.title('Values')
​
plt.show()
Creating comparisons with bar charts
import matplotlib.pyplot as plt
%matplotlib inline
​
values = [5, 8, 9, 10, 4, 7]
widths = [0.7, 0.8, 0.7, 0.7, 0.7, 0.7]
colors = ['b', 'r', 'b', 'b', 'b', 'b']
plt.bar(range(0, 6), values, width=widths, 
        color=colors, align='center')
​
plt.show()
Showing distributions using histograms
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
​
x = 20 * np.random.randn(10000)
​
plt.hist(x, 25, range=(-50, 50), histtype='stepfilled',
         align='mid', color='g', label='Test Data')
plt.legend()
plt.title('Step Filled Histogram')
plt.show()
Depicting groups using box plots
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
​
spread = 100 * np.random.rand(100)
center = np.ones(50) * 50
flier_high = 100 * np.random.rand(10) + 100
flier_low = -100 * np.random.rand(10)
data = np.concatenate((spread, center, 
                       flier_high, flier_low))
​
plt.boxplot(data, sym='gx', widths=.75, notch=True)
plt.show()
Seeing data patterns using scatterplots
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
​
x1 = 5 * np.random.rand(40)
x2 = 5 * np.random.rand(40) + 25
x3 = 25 * np.random.rand(20)
x = np.concatenate((x1, x2, x3))
​
y1 = 5 * np.random.rand(40)
y2 = 5 * np.random.rand(40) + 25
y3 = 25 * np.random.rand(20)
y = np.concatenate((y1, y2, y3))
​
plt.scatter(x, y, s=[100], marker='^', c='m')
plt.show()
Creating Advanced Scatterplots
Depicting groups
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
​
x1 = 5 * np.random.rand(50)
x2 = 5 * np.random.rand(50) + 25
x3 = 30 * np.random.rand(25)
x = np.concatenate((x1, x2, x3))
​
y1 = 5 * np.random.rand(50)
y2 = 5 * np.random.rand(50) + 25
y3 = 30 * np.random.rand(25)
y = np.concatenate((y1, y2, y3))
​
color_array = ['b'] * 50 + ['g'] * 50 + ['r'] * 25
​
plt.scatter(x, y, s=[50], marker='D', c=color_array)
plt.show()
#Showing correlations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
#%matplotlib inline
​
x1 = 15 * np.random.rand(50)
x2 = 15 * np.random.rand(50) + 15
x3 = 30 * np.random.rand(25)
x = np.concatenate((x1, x2, x3))
​
y1 = 15 * np.random.rand(50)
y2 = 15 * np.random.rand(50) + 15
y3 = 30 * np.random.rand(25)
y = np.concatenate((y1, y2, y3))
​
color_array = ['b'] * 50 + ['g'] * 50 + ['r'] * 25
plt.scatter(x, y, s=[90], marker='*', c=color_array)
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plb.plot(x, p(x), 'm-')
​
plt.show()
#Plotting Time Series
#Representing time on axes

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
#%matplotlib inline
​
start_date = dt.datetime(2018, 7, 29)
end_date = dt.datetime(2018, 8, 7)
daterange = pd.date_range(start_date, end_date)
sales = (np.random.rand(len(daterange)) * 50).astype(int)
df = pd.DataFrame(sales, index=daterange, 
                  columns=['Sales']) 
​
df.loc['Jul 30 2018':'Aug 05 2018'].plot()
plt.ylim(0, 50)
plt.xlabel('Sales Date')
plt.ylabel('Sale Value')
plt.title('Plotting Time')
plt.show()


#Plotting trends over time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
#%matplotlib inline
​
start_date = dt.datetime(2018, 7, 29)
end_date = dt.datetime(2018, 8, 7)
daterange = pd.date_range(start_date, end_date)
sales = (np.random.rand(len(daterange)) * 50).astype(int)
df = pd.DataFrame(sales, index=daterange, 
                  columns=['Sales']) 
​
lr_coef = np.polyfit(range(0, len(df)), df['Sales'], 1)
lr_func = np.poly1d(lr_coef)
trend = lr_func(range(0, len(df)))
df['trend'] = trend
df.loc['Jul 30 2018':'Aug 05 2018'].plot()
​
plt.xlabel('Sales Date')
plt.ylabel('Sale Value')
plt.title('Plotting Time')
plt.legend(['Sales', 'Trend'])
plt.show()
Plotting Geographical Data
# In order to run this code you have use conda from the Anaconda Prompt
# to install these packages:
​
# conda install -c conda-forge basemap=1.1.0
# conda install -c conda-forge basemap-data-hires
# conda install -c conda-forge proj4=5.2.0
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
%matplotlib inline
import warnings

# ^ Importing Basemap for Geographic Data

warnings.filterwarnings("ignore")
austin = (-97.75, 30.25)
hawaii = (-157.8, 21.3)
washington = (-77.01, 38.90)
chicago = (-87.68, 41.83)
losangeles = (-118.25, 34.05)
​
​
m = Basemap(projection='merc',llcrnrlat=10,urcrnrlat=50,
            llcrnrlon=-160,urcrnrlon=-60)
​
m.drawcoastlines()
m.fillcontinents(color='lightgray',lake_color='lightblue')
m.drawparallels(np.arange(-90.,91.,30.))
m.drawmeridians(np.arange(-180.,181.,60.))
m.drawmapboundary(fill_color='aqua')
​
m.drawcountries()
​
x, y = m(*zip(*[hawaii, austin, washington, 
                chicago, losangeles]))
m.plot(x, y, marker='o', markersize=6, 
       markerfacecolor='red', linewidth=0)
​
plt.title("Mercator Projection")
plt.show()
Visualizing Graphs
Developing undirected graphs
import networkx as nx

# ^ Importation of NetworkX

# NetworkX is used for node-graphs, or arrays placed in diretional plots

#Pursuing this study going forward to find GeoThermal data
#has to be layered with geolocale by sattelite imagery

import matplotlib.pyplot as plt
%matplotlib inline
​
G = nx.Graph()
H = nx.Graph()
G.add_node(1)
G.add_nodes_from([2, 3])
G.add_nodes_from(range(4, 7))
H.add_node(7)
G.add_nodes_from(H)
​
G.add_edge(1, 2)
G.add_edge(1, 1)
G.add_edges_from([(2,3), (3,6), (4,6), (5,6)])
H.add_edges_from([(4,7), (5,7), (6,7)])
G.add_edges_from(H.edges())
​
nx.draw_networkx(G)
plt.show()
Developing directed graphs
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline
​
G = nx.DiGraph()
​
G.add_node(1)
G.add_nodes_from([2, 3])
G.add_nodes_from(range(4, 6))
G.add_path([6, 7, 8])
​
G.add_edge(1, 2)
G.add_edges_from([(1,4), (4,5), (2,3), (3,6), (5,6)])
​
colors = ['r', 'g', 'g', 'g', 'g', 'm', 'm', 'r']
labels = {1:'Start', 2:'2', 3:'3', 4:'4', 
          5:'5', 6:'6', 7:'7', 8:'End'}
sizes = [800, 300, 300, 300, 300, 600, 300, 800]
​
nx.draw_networkx(G, node_color=colors, node_shape='D', 
                 with_labels=True, labels=labels, 
                 node_size=sizes)
plt.show()

