import streamlit as st
import itertools
import matplotlib.colors
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
        
st.markdown('<p class="maintitle">Exploring ERGM</p>', unsafe_allow_html=True)
    
