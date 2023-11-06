import streamlit as st
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

from matplotlib.gridspec import GridSpec
from mpl_toolkits import mplot3d 

def get_vertices(graph):
    # Not sensible, but a good test
    return len(graph.nodes())

def get_edges(graph):
    return len(graph.edges())

def get_triangles(graph):
    # Returns the number of vertices in triangles
    return sum(nx.triangles(graph).values())

def get_isolates(graph):
    # Vertices without any edges to other vertices
    return len(list(nx.isolates(graph)))

def get_ergm_weight(graph, coefficients, statistics):
    '''
    Compute the ERGM numerator on graph with coefficients for a set of graph statistics    
    '''
    v = 0
    for i,stat in enumerate(statistics):
        v += coefficients[i] * stat(graph)        
    return np.exp(v)

def get_ergm_denominator(graph_set, coefficients, statistics):
    denom = 0
    for graph in graph_set:
        denom += get_ergm_weight(graph, coefficients, statistics)
    return denom
    
def coeffs_to_string(coefficients):
    s = ''
    for i,c in enumerate(coefficients):
        s += str(c)
        if i < len(coefficients) - 1:
            s += ', '
    return s

def pr_analysis(coeff_values,non_zero_coeff_values,statistics,label):    
    pr_dict = {}
    for t in coeff_values:
        # Other coefficients are 0
        coefficients = [t[0],t[1],t[2]]
        denom = get_ergm_denominator(graph_set, coefficients, statistics)

        for i,g in enumerate(graph_set):
            numerator = get_ergm_weight(g, coefficients, statistics)
            pr = round(numerator/denom,3)
            if i in pr_dict:
                pr_dict[i].append(pr)
            else:
                pr_dict[i] = [pr]
    fig = plt.figure(figsize=(8,4),layout="constrained")
    x = non_zero_coeff_values
    for k,v in pr_dict.items():
        plt.plot(x,v,label=str(k))
        plt.scatter(x,v)
    #plt.ylim(0,1)
    plt.xlabel('Value of ' + label + ' statistic coefficient')
    plt.ylabel('Pr of graph')
    plt.legend()
    plt.title(label)
    st.pyplot(fig)

        
st.header('Exploring ERGMs')

st.subheader('Introduction')

st.write('This app is an attempt to understand and explore ERGMs using a set of 3-node undirected graphs as an example. This set has been chosen because it\'s tractable and can be visualised.')

st.write('The probability of observing a graph $g$ that belongs to a set of graphs $G$ is given by:')

st.latex(r'''
    pr(g|G) = \frac{\exp(a_1s_1 + a_2s_2 + … + a_Ns_N)}{\sum\limits_{g=1}^{|G|}\exp(a_1s_1 + a_2s_2 + … + a_Ns_N)}
    ''')

st.write('The linear equation:')

st.latex(r'''
    a_1s_1 + a_2s_2 + … + a_Ns_N)
    ''')

st.write('Combines a set of network statistics ${s_1,s_2,…,s_N}$ in proportions determined by set of coefficients ${a_1,a_2,…,a_N}$')

st.write('If all coefficient values are set to zero then:')

st.latex(r'''
    pr(g|G) = \frac{e^0}{\sum\limits_{g=1}^{|G|}e^0)} = \frac{1}{|G|}
    ''')

   
# Build the set of three node graphs as a sample set for denominator calculation
# This needs to be sensible

graph_set = []
g = nx.Graph()
g.add_nodes_from([0,1,2])
graph_set.append(g)

g = nx.Graph()
g.add_nodes_from([0,1,2])
g.add_edge(0,1)
graph_set.append(g)

g = nx.Graph()
g.add_nodes_from([0,1,2])
g.add_edge(0,2)
graph_set.append(g)

g = nx.Graph()
g.add_nodes_from([0,1,2])
g.add_edge(1,2)
graph_set.append(g)

g = nx.Graph()
g.add_nodes_from([0,1,2])
g.add_edge(0,1)
g.add_edge(0,2)
graph_set.append(g)

g = nx.Graph()
g.add_nodes_from([0,1,2])
g.add_edge(0,1)
g.add_edge(1,2)
graph_set.append(g)

g = nx.Graph()
g.add_nodes_from([0,1,2])
g.add_edge(0,2)
g.add_edge(1,2)
graph_set.append(g)

g = nx.Graph()
g.add_nodes_from([0,1,2])
g.add_edge(0,1)
g.add_edge(0,2)
g.add_edge(1,2)
graph_set.append(g)

st.subheader('The set of 3-node undirected graphs')
st.write('Graph titles include the values of the three network statistics used in this analysis:')
st.write('- Number of edges')
st.write('- Number of isolates')
st.write('- Number of triangles')


fig = plt.figure(figsize=(8,4),layout="constrained")

gs = GridSpec(2, 4, figure=fig)

for i,g in enumerate(graph_set):
    ax = fig.add_subplot(gs[i//4,i%4])
    nx.draw_networkx(g,node_color='pink')
    title = 'Graph ' + str(i) + ': ' + str(get_edges(g)) + ', ' + str(get_isolates(g)) + \
        ', ' + str(get_triangles(g))
    ax.set_title(title)
st.pyplot(fig)


with st.form("coefficients"):
    st.subheader('Probability of observing a graph from the set of 3-node graphs given some coefficient values')
    st.write('Use the form below to set the coefficient values for three network statistics:')   
    st.write('Number of edges')
    st.write('Number of isolates')
    st.write('Number of triangles')
    st.write('In the default condition all coefficient values are set to zero and all graphs in the set are equally likely to be observed, i.e., $Pr(g|G)=\frac{1}{8}=0.125$')
   
    statistics = [get_edges,get_isolates,get_triangles]

    edges=st.slider("Select edges coefficient", min_value=0, max_value=1, value=0, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    isolates=st.slider("Select isolates coefficient", min_value=0, max_value=1, value=0, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    triangles=st.slider("Select triangles coefficient", min_value=0, max_value=1, value=0, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

# Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:

        coefficients = [edges,isolates,triangles]
        fig = plt.figure(figsize=(8,4),layout="constrained")

        gs = GridSpec(2, 4, figure=fig)

        denom = get_ergm_denominator(graph_set, coefficients, statistics)

        for i,g in enumerate(graph_set):
            numerator = get_ergm_weight(g, coefficients, statistics)
            pr = round(numerator/denom,3)

            ax = fig.add_subplot(gs[i//4,i%4])
            nx.draw_networkx(g,node_color='pink')
            title = 'Graph ' + str(i) + ' Pr = ' + str(pr)
            ax.set_title(title)
            
        st.pyplot(fig)

st.subheader('Effects of coefficient of single statistic on probability of observing a graph.')
st.write('The coefficients of the other two statistics are set to zero.')


statistics = [get_edges,get_isolates,get_triangles]

non_zero_coeff_values = np.arange(0,1.1,0.1)
zero_values = np.zeros(len(non_zero_coeff_values))

coeff_values = list(zip(non_zero_coeff_values,zero_values,zero_values))
pr_analysis(coeff_values,non_zero_coeff_values,statistics,'Edges')

coeff_values = list(zip(zero_values,non_zero_coeff_values,zero_values))
pr_analysis(coeff_values,non_zero_coeff_values,statistics,'Isolates')

coeff_values = list(zip(zero_values,zero_values,non_zero_coeff_values))
pr_analysis(coeff_values,non_zero_coeff_values,statistics,'Triangles')


st.subheader('Joint distributions for edge and triangle statistics for graph 7.')

statistics = [get_edges,get_isolates,get_triangles]

g = graph_set[7]


non_zero_coeff_values = np.arange(0,1.1,0.1)

matrix = np.zeros((len(non_zero_coeff_values),len(non_zero_coeff_values)))


for i,x in enumerate(non_zero_coeff_values):
    for j,y in enumerate(non_zero_coeff_values):
        coefficients = [x,0,y]
        denom = get_ergm_denominator(graph_set, coefficients, statistics)
        numerator = get_ergm_weight(g, coefficients, statistics)
        pr = round(numerator/denom,3)
        matrix[i,j] = pr


fig = plt.figure(figsize=(8, 8))
plt.imshow(matrix)
plt.colorbar()
plt.ylabel('Edges coefficent')
plt.xlabel('Triangles coefficent')
#plt.clim(0,1)

st.pyplot(fig)

m,n = matrix.shape
fig = plt.figure(figsize=(16, 8))
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(non_zero_coeff_values, non_zero_coeff_values)
p = ax.plot_surface(X, Y, matrix, rstride=1, cstride=1, cmap='viridis', edgecolor='none',\
                    antialiased=False,vmin=0,vmax=1.0)
ax.set_zlim(0,1)
ax.set_xlabel('Edges coefficient')
ax.set_ylabel('Triangles coefficient')
ax.set_zlabel('Probability of graph')
#cbar = fig.colorbar(p,ax=ax)
#cbar.set_label('Probability of graph', rotation=270)

st.pyplot(fig)


st.subheader('Joint distributions for edge and triangle statistics for all graphs.')

statistics = [get_edges,get_isolates,get_triangles]
non_zero_coeff_values = np.arange(0,1.1,0.1)

fig = plt.figure(figsize=(20,20))

gs = GridSpec(4, 2, figure=fig)

for k,g in enumerate(graph_set):

    matrix = np.zeros((len(non_zero_coeff_values),len(non_zero_coeff_values)))


    for i,x in enumerate(non_zero_coeff_values):
        for j,y in enumerate(non_zero_coeff_values):
            coefficients = [x,0,y]
            denom = get_ergm_denominator(graph_set, coefficients, statistics)
            numerator = get_ergm_weight(g, coefficients, statistics)
            pr = round(numerator/denom,3)
            matrix[i,j] = pr

    m,n = matrix.shape
    ax = fig.add_subplot(gs[k%4,k//4],projection='3d')

    #ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(non_zero_coeff_values, non_zero_coeff_values)
    p = ax.plot_surface(X, Y, matrix, rstride=1, cstride=1, cmap='viridis', edgecolor='none',\
                        antialiased=False,vmin=0,vmax=1.0)
    ax.set_zlim(0,1)
    ax.set_xlabel('Edges coefficient')
    ax.set_ylabel('Triangles coefficient')
    ax.set_zlabel('Probability of graph')
    ax.set_title('Graph: ' + str(k))
    #cbar = fig.colorbar(p,ax=ax)
    #cbar.set_label('Probability of graph', rotation=270)

st.pyplot(fig)


