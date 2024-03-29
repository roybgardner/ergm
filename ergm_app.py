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
    # Returns the number of triangles
    return sum(nx.triangles(graph).values())/3

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
        
st.header('Exploring ERGMs')

st.subheader('Introduction')

st.write('This app is an attempt to understand and explore ERGMs using a set of 3-vertex undirected graphs as an example. This set has been chosen because it\'s tractable and can be visualised.\
         There are eight graphs in the set (see below). As the number of vertices increases the number of graphs grows rapidly to the point where the denominator in the equation below cannot be computed.\
         The number of graphs is $2^{(n\\times(n-1))/2}$ where $n$ is the number of vertices.')

st.write('The probability of observing a graph $g^\prime$ that belongs to a set of graphs $G$ is given by:')

st.latex(r'''
    pr(g^\prime|G) = \frac{\exp(a_1s(g^\prime)_1 + a_2s(g^\prime)_2 + … + a_Ns(g^\prime)_N)}{\sum\limits_{g=1}^{|G|}\exp(a_1s(g)_1 + a_2s(g)_2 + … + a_Ns(g)_N)}
    ''')

st.write('The linear equation:')

st.latex(r'''
    a_1s(g)_1 + a_2s(g)_2 + … + a_Ns(g)_N
    ''')

st.write('combines a set of network statistics $\{s(g)_1,s(g)_2,…,s(g)_N\}$ of a graph $g$ in proportions determined by a set of coefficients $\{a_1,a_2,…,a_N\}$')

st.write('If all coefficient values are set to zero then all graphs are equally likely:')

st.latex(r'''
    pr(g^\prime|G) = \frac{e^0}{\sum\limits_{g=1}^{|G|}e^0} = \frac{1}{|G|}
    ''')

st.subheader('The set of 3-vertex undirected graphs')
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
        ', ' + str(int(get_triangles(g)))
    ax.set_title(title)
st.pyplot(fig)


with st.form("coefficients"):
    st.subheader('Probability of observing a graph from the set of 3-vertex graphs given some coefficient values')
    st.write('Use the form below to set the coefficient values for the three network statistics.')   
    st.write('In the default condition, all coefficient values are set to zero and all graphs in the set are equally likely to be observed, i.e., $Pr(g|G)=0.125$')
   
    statistics = [get_edges,get_isolates,get_triangles]

    edges=st.slider("Select edges coefficient", min_value=0.0, max_value=1.0, value=0.0, step=0.1, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    isolates=st.slider("Select isolates coefficient", min_value=0.0, max_value=1.0, value=0.0, step=0.1, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")
    triangles=st.slider("Select triangles coefficient", min_value=0.0, max_value=1.0, value=0.0, step=0.1, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

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

st.subheader('Effects of coefficient of a single statistic on the probability of observing a graph.')
st.write('The coefficients of the remaining statistics are set to zero.')


statistics = [get_edges,get_isolates,get_triangles]

non_zero_coeff_values = np.arange(0,1.1,0.1)
zero_values = np.zeros(len(non_zero_coeff_values))

coeff_values = list(zip(non_zero_coeff_values,zero_values,zero_values))
pr_analysis(coeff_values,non_zero_coeff_values,statistics,'Edges')

coeff_values = list(zip(zero_values,non_zero_coeff_values,zero_values))
pr_analysis(coeff_values,non_zero_coeff_values,statistics,'Isolates')

coeff_values = list(zip(zero_values,zero_values,non_zero_coeff_values))
pr_analysis(coeff_values,non_zero_coeff_values,statistics,'Triangles')


with st.form("joint_7"):
    st.subheader('Joint distributions of pairs of statistics for graph 7.')


    joint_options=["Edges-Triangles", "Edges-Isolates", "Isolates-Triangles"]
    selected_joint=st.radio("Select pair", joint_options, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, horizontal=False, captions=None, label_visibility="visible")

    submitted = st.form_submit_button("Submit")
    if submitted:
        statistics = [get_edges,get_isolates,get_triangles]
        g = graph_set[7]
        non_zero_coeff_values = np.arange(0,1.1,0.1)

        zero_values = np.zeros(len(non_zero_coeff_values))

        matrix = np.zeros((len(non_zero_coeff_values),len(non_zero_coeff_values)))
        for i,x in enumerate(non_zero_coeff_values):
            for j,y in enumerate(non_zero_coeff_values):
                if selected_joint == "Edges-Triangles":
                    coefficients = [x,0,y]
                elif selected_joint == "Edges-Isolates":
                    coefficients = [x,y,0]
                elif selected_joint == "Isolates-Triangles":
                    coefficients = [0,x,y]

                denom = get_ergm_denominator(graph_set, coefficients, statistics)
                numerator = get_ergm_weight(g, coefficients, statistics)
                pr = round(numerator/denom,3)
                matrix[i,j] = pr


        fig = plt.figure(figsize=(8, 8))
        tick_labels = [round(v,1) for v in non_zero_coeff_values]
        plt.imshow(matrix)
        plt.colorbar()
        plt.yticks(range(0,len(non_zero_coeff_values)),tick_labels)
        plt.xticks(range(0,len(non_zero_coeff_values)),tick_labels)
        plt.ylabel(selected_joint.split('-')[0] + ' coefficient')
        plt.xlabel(selected_joint.split('-')[1] + ' coefficient')
        #plt.clim(0,1)

        st.pyplot(fig)

with st.form("joint_all"):


    st.subheader('Joint distributions of pairs of statistics for all graphs.')
    st.write('Note that the colorbar range is subplot specific.')

    joint_options=["Edges-Triangles", "Edges-Isolates", "Isolates-Triangles"]
    selected_joint=st.radio("Select pair", joint_options, index=0, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, horizontal=False, captions=None, label_visibility="visible")


    submitted = st.form_submit_button("Submit")
    if submitted:
        statistics = [get_edges,get_isolates,get_triangles]
        non_zero_coeff_values = np.arange(0,1.1,0.1)
        
        fig = plt.figure(figsize=(20,20))
        tick_labels = [round(v,1) for v in non_zero_coeff_values]

        gs = GridSpec(4, 2, figure=fig)

        for k,g in enumerate(graph_set):

            matrix = np.zeros((len(non_zero_coeff_values),len(non_zero_coeff_values)))


            for i,x in enumerate(non_zero_coeff_values):
                for j,y in enumerate(non_zero_coeff_values):
                    if selected_joint == "Edges-Triangles":
                        coefficients = [x,0,y]
                    elif selected_joint == "Edges-Isolates":
                        coefficients = [x,y,0]
                    elif selected_joint == "Isolates-Triangles":
                        coefficients = [0,x,y]
                    denom = get_ergm_denominator(graph_set, coefficients, statistics)
                    numerator = get_ergm_weight(g, coefficients, statistics)
                    pr = round(numerator/denom,3)
                    matrix[i,j] = pr

            ax = fig.add_subplot(gs[k%4,k//4])
            
            im = ax.imshow(matrix)
            ax.set_yticks(range(0,len(non_zero_coeff_values)),tick_labels)
            ax.set_xticks(range(0,len(non_zero_coeff_values)),tick_labels)
            ax.set_ylabel(selected_joint.split('-')[0] + ' coefficient')
            ax.set_xlabel(selected_joint.split('-')[1] + ' coefficient')
            ax.set_title('Graph ' + str(k))
            fig.colorbar(im, ax=ax)

        st.pyplot(fig)


