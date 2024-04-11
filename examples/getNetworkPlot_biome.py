#! /usr/bin/env python

#from __future__ import division, absolute_import, print_function
import sys
import os
import argparse
from matplotlib import cm
import graph_tool.all as gt
import numpy as np
from graph_tool.topology import extract_largest_component
import cairo

def is_causal_cycle(cycle_graph):
    vertex_name_prop = cycle_graph.vertex_properties['vertex_name']
    # Function to extract the timestamp from the vertex name
    def get_timestamp(vertex_name):
        _, timestamp_str = vertex_name.rsplit('_', 1)
        return int(timestamp_str)

    # Iterate through each edge in the cycle
    for edge in cycle_graph.edges():
        src = edge.source()
        tgt = edge.target()

        src_timestamp = get_timestamp(vertex_name_prop[src])
        tgt_timestamp = get_timestamp(vertex_name_prop[tgt])

        print( tgt_timestamp , src_timestamp)

        # Check if the target timestamp is not more than 2 less than the source timestamp
        if tgt_timestamp < src_timestamp - 50:
            return False  # The cycle is not causal

    return True  # The cycle is causal

def extract_cycle_of_length(g, L):
    for circuit in gt.all_circuits(g):
        if len(circuit) >= L:
            print(circuit)
            # Get the vertex and edge sets for the cycle
            vertex_set = set(circuit)
            edge_set = set()
            for i in range(len(circuit)):
                start_vertex = circuit[i]
                end_vertex = circuit[(i + 1) % len(circuit)]
                edge_set.add(g.edge(start_vertex, end_vertex))

            # Create a GraphView based on the vertex and edge sets
            cycle_graph_view = gt.GraphView(g, vfilt=lambda v: v in vertex_set,
                                            efilt=lambda e: e in edge_set)

            # Optionally, create a separate graph based on the view if a standalone graph is needed
            cycle_graph = gt.Graph(cycle_graph_view, prune=True)

            if is_causal_cycle(cycle_graph):
                return cycle_graph
            else:
                print('nonausal')

    return None  # Return None if no cycle of the desired length is found

#ADD=3
#EXP=False
DOTNAME='covid.dot'
ADD=0
EXP=True
color_map='Spectral_r'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#--------------------------------------------
def f(x,A=0,E=True,exponent=2.0):
    if E:
        return exponent**x + A
    return x+A

USAGE='./getNetworkPlot.py covid.dot -o covid.pdf  -v True -a 0.8 -c Spectral_r'
parser = argparse.ArgumentParser(description='----| Network Visualizer with ARF layout |----')
parser.add_argument('DOTNAME', nargs=1, type=str,
                    help="dotfile")
parser.add_argument('-outfile', dest='OUTFILE', action="store", type=str,
                    default='out.pdf',help="output file (default: %(default)s)")
parser.add_argument('-EXP', dest='EXP', action="store", type=str2bool,
                    default=True,help="exponential scaling of node sie with out degree (default: %(default)s)")
parser.add_argument('-ADD', dest='ADD', action="store", type=int,
                    default=0,help="min node size (default: %(default)s)")
parser.add_argument('-alpha', dest='ALPHA', action="store", type=float,
                    default=0.7,help="alpha value (default: %(default)s)")
parser.add_argument('-exp', dest='exponent', action="store", type=float,
                    default=2.0,help="exponent value (default: %(default)s)")
parser.add_argument('-cmap', dest='COLMAP', action="store", type=str,
                    default='hot',help="colormap (default: %(default)s)")
parser.add_argument('-verbose', dest='VERBOSE', action="store", type=str2bool,
                    default=False,help="verbose default: %(default)s)")

args_=parser.parse_args()
EXP=args_.EXP
ADD=args_.ADD
DOTNAME=args_.DOTNAME[0]
OUTFILE=args_.OUTFILE

if args_.VERBOSE:
    print(args_)

CMAP=cm.get_cmap(args_.COLMAP)
ALPHA=args_.ALPHA

#--------------------------------------------
g = gt.load_graph(DOTNAME)
#g = extract_largest_component(g, directed=True, prune=True)
#L = 3
#g = extract_cycle_of_length(g, L)


if args_.VERBOSE:
    for v in g.vertices():
        print('deg',v.out_degree())
    for v in g.vertices():
        print('vertex_name',g.vp.vertex_name[v])
    g.list_properties()
    

nm = g.new_vertex_property("string")
hl = g.new_vertex_property("bool")

g.vertex_properties["text"] = nm
for v in g.vertices():
    #print(v.out_degree() )
    if v.out_degree() > 2:
        nm[v]=g.vp.vertex_name[v].replace('P','')
        nm[v] = nm[v][:5]+nm[v][-3:]
        #print(nm[v])
        hl[v]=True
    else:
        #nm[v]=''
        hl[v]=False
        
od = g.new_vertex_property("float")
g.vertex_properties["size"] = od
for v in g.vertices():
    od[v]=f(v.out_degree(),A=ADD,E=EXP,exponent=args_.exponent)+5

ew = g.new_edge_property("float")
g.edge_properties["eweight"] = ew
for e in g.edges():
    ew[e]=float(g.ep.weight[e])**2

deg = g.degree_property_map("out")
pos = gt.arf_layout(g,weight=g.ep.eweight,a=12,max_iter=500,d=5)

control = g.new_edge_property("vector<double>")
for e in g.edges():
    d = np.sqrt(np.sum((pos[e.source()].a - pos[e.target()].a) ** 2)) / 5
    control[e] = [0.1, d, 0.5, d]

gt.graph_draw(g,nodesfirst=False,
              pos=pos,
              vertex_halo=hl,
              vertex_halo_color=[.2,.2,.2,.05],
              edge_pen_width=1.5,
              vorder=deg,
              edge_marker_size=8,
              vertex_color=[.15,.15,.15,.9],
              edge_color=[.5,.5,.5,.3],
              vertex_pen_width=2,
              vertex_size=od,
              vertex_text=nm,
              vcmap=(CMAP,ALPHA),#edge_control_points=control,
              vertex_fill_color=deg,vertex_font_size=10,
              vertex_font_weight=cairo.FONT_WEIGHT_BOLD,
              output=OUTFILE)

