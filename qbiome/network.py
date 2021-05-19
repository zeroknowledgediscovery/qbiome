import pandas as pd
import graph_tool.all as gt
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm


class Network(object):
    
    def __init__(self,
                 input_dotfile,
                 cmap=cm.Spectral_r,
                 edgecolmap=None,
                 preflen=8,
                 alpha=.7,
                 minsize=5,
                 edgealpha=.2,
                 exponent=1,
                 edgecollim=2,
                 strongcomponent=True,
                 removeselfloops=True,
                 exponentialscaling=True,
                 outfile=None):
        """Process hypothesis generated dotfile for network visualization

        Args:
          input_dotfile (str): dotfile generated from hypothesis module
          cmap (matplotlib colormap):  color map for nodes (Default value = cm.Spectral_r)
          edgecolmap (matplotlib colormap):  edge color map (Default value = None)
          preflen (int):  max length of node names (Default value = None)
          alpha (float):  alpha for nodes (Default value = .7)
          minsize (int):  minimum soze of nodes (Default value = 5)
          edgealpha (float):  edge alpha (Default value = .2)
          exponent (int):  exponent in exponential scaling (Default value = 1)
          edgecollim (int):  range for colomap on edges (Default value = 2)
          strongcomponent (bool):  if True only show strong component (Default value = True)
          removeselfloops (bool):  if True remove self loops (Default value = True)
          exponentialscaling (bool):  if True use exponential scaling (Default value = True)
          outfile (str):  output filename (Default value = None)

        Returns:

        """
        if edgecolmap is None:
            self.edgecolmap=LinearSegmentedColormap.from_list(
                'edgecolmap',
                ['#0000ff',
                 '#888888',
                 '#ff0000'])
        else:
            self.edgecolmap=edgecolmap
        self.minsize=minsize
        self.cmap=cmap
        self.network=None
        self.dotfile=input_dotfile
        self.preflen=preflen
        self.edgealpha=edgealpha
        self.exponent=exponent
        self.edgecollim=edgecollim
        self.strongcomponent=strongcomponent
        self.removeselfloops=removeselfloops
        self.exponentialscaling=exponentialscaling
        self.alpha=alpha
        self.nm=None
        self.od=None
        self.pos=None
        self.hl=None
        self.ew_pen=None
        self.e_marker=None
        self.deg=None
        self.control=None
        self.ods=None
        self.varclass=None
        self.ecol=None
        self.outfile=outfile
        
    def short_name(self,s,LEN):
        """

        Args:
          s (str): string
          LEN (nt): length of short name

        Returns:
          str: short name

        """
        num=len(s.split('_'))
        if num>1:
            LEN=int(LEN//num) + 1
            
        return '_'.join([x[:LEN] for x in s.split('_')])
        
        
    def f(self,x,A=0,E=True,exponent=2.0):
        """adjust node sizes

        Args:
          x: 
          A:  (Default value = 0)
          E:  (Default value = True)
          exponent:  (Default value = 2.0)

        Returns:
          float
        """
        if E:
            return exponent**x + A
        return x+A

    def sfunc(self,val,SIGN=False):
        """

        Args:
          val: 
          SIGN:  (Default value = False)

        Returns:
          float
        """
        if SIGN:
            return np.sign(val)
        return val

 
    def get(self):
        """Set graph-tool network for drawing"""
        self.network = gt.load_graph(self.dotfile)

        if self.strongcomponent:
            self.network=gt.extract_largest_component(
                self.network, directed=True, prune=True)

        if self.removeselfloops:
            gt.remove_self_loops(self.network)

        self.nm = self.network.new_vertex_property("string")
        nm2 = self.network.new_vertex_property("string")
        self.hl = self.network.new_vertex_property("bool")
        self.network.vertex_properties["text"] = self.nm
        self.network.vertex_properties["text"] = nm2
        names=[]
        for v in self.network.vertices():
            if v.out_degree() > -1:
                self.nm[v]=self.short_name(
                    self.network.vp.vertex_name[v],self.preflen)
                nm2[v]=self.short_name(
                    self.network.vp.vertex_name[v],self.preflen)
                self.hl[v]=False
            else:
                nm2[v]=self.short_name(
                    self.network.vp.vertex_name[v],self.preflen)
                self.nm[v]=''
                self.hl[v]=False
            names=names+[nm2[v]]

        NAMES=pd.Series(list(set(names)),
                        name='varclass').reset_index().set_index('varclass')
        self.varclass = self.network.new_vertex_property("float")
        self.network.vertex_properties["varclass"] = self.varclass
        for v in self.network.vertices():
            self.varclass[v]=NAMES.loc[nm2[v]].values[0]

        self.od = self.network.new_vertex_property("float")
        self.network.vertex_properties["size"] = self.od
        for v in self.network.vertices():
            self.od[v]=self.f(v.out_degree(),
                              A=self.minsize,
                              E=self.exponentialscaling,
                              exponent=self.exponent)+5
        self.ods = self.network.new_vertex_property("float")
        self.network.vertex_properties["size"] = self.ods
        for v in self.network.vertices():
            self.ods[v]=1*self.f(v.out_degree(),
                                 A=self.minsize,
                                 E=self.exponentialscaling,
                                 exponent=1)+2

        self.ew = self.network.new_edge_property("float")
        self.network.edge_properties["eweight"] = self.ew
        for e in self.network.edges():
            self.ew[e]=float(self.network.ep.weight[e])**1

        self.ew_pen = self.network.new_edge_property("float")
        self.network.edge_properties["eweight_pen"] = self.ew_pen
        for e in self.network.edges():
            self.ew_pen[e]=4/(1 + np.exp(-.05-np.fabs(float(self.network.ep.weight[e]))))

        self.e_marker = self.network.new_edge_property("string")
        self.network.edge_properties["e_marker"] = self.e_marker
        for e in self.network.edges():
            if float(self.network.ep.weight[e]) < 0:
                self.e_marker[e]='bar'
            else:
                self.e_marker[e]='arrow'

        self.deg = self.network.degree_property_map("out")

        self.ecol = self.network.new_edge_property("vector<double>")
        self.network.edge_properties["ecol"] = self.ecol
        for e in self.network.edges():
            col=cm.ScalarMappable(mpl.colors.Normalize(vmin=-self.edgecollim,
                                                       vmax=self.edgecollim),
                                  cmap=self.edgecolmap).to_rgba(float(self.ew[e]))
            col=list(col)
            col[3]=self.edgealpha
            self.ecol[e]=tuple(col)

        self.pos = gt.graphviz_draw(self.network,
                                    overlap=False,
                                    vsize=20,
                                    sep=20,
                                    output=None)

        self.control = self.network.new_edge_property("vector<double>")
        for e in self.network.edges():
            d = np.sqrt(np.sum((self.pos[e.source()].a
                                - self.pos[e.target()].a) ** 2))
            d=d/2
            self.control[e] = [0.0,0.0,0, .2*d, 0.5, d,1,0]

        if self.outfile is not None:
            gt.graph_draw(self.network,nodesfirst=False,
                          pos=self.pos,
                          vertex_halo=self.hl,
                          vertex_halo_color=[.2,.2,.2,.1],
                          edge_pen_width=self.ew_pen,
                          edge_end_marker=self.e_marker,
                          vorder=self.deg,
                          edge_marker_size=10,
                          vertex_color=self.varclass,#[.5,.5,.5,.3],
                          edge_color=self.ecol,#[.5,.5,.5,.5],
                          vertex_pen_width=1.5,
                          vertex_size=self.od,
                          vertex_text=self.nm,
                          vcmap=(self.cmap,self.alpha),
                          edge_control_points=self.control,
                          vertex_fill_color=self.varclass,#deg,
                          vertex_font_size=self.ods,
                          vertex_text_color=[.1,.1,.1,.8],
                          #vertex_text_position=0,
                          output=self.outfile)

    def draw(self,
             pen_width=None,
             cmap=None,
             alpha=None,
             text_pos=None):
        """Draw  network

        Args:
          pen_width (float):  (Default value = None)
          cmap (matplotlib colormap):  (Default value = None)
          alpha (float):  (Default value = None)
          text_pos (float): if not None, puts node text below node (Default value = None)

        Returns:

        """

        if pen_width is None:
            edge_pen_width=self.ew_pen
        else:
            edge_pen_width=pen_width
            
        if cmap is None:
            cmap=self.cmap
            
        if alpha is None:
            alpha=self.alpha

        if text_pos is None:
            gt.graph_draw(self.network,nodesfirst=False,
                          pos=self.pos,
                          vertex_halo=self.hl,
                          vertex_halo_color=[.2,.2,.2,.1],
                          edge_pen_width=pen_width,
                          edge_end_marker=self.e_marker,
                          vorder=self.deg,
                          edge_marker_size=10,
                          vertex_color=self.varclass,#[.5,.5,.5,.3],
                          edge_color=self.ecol,#[.5,.5,.5,.5],
                          vertex_pen_width=1.5,
                          vertex_size=self.od,
                          vertex_text=self.nm,
                          vcmap=(cmap,alpha),
                          edge_control_points=self.control,
                          vertex_fill_color=self.varclass,#deg,
                          vertex_font_size=self.ods,
                          vertex_text_color=[.1,.1,.1,.8],
                          output=self.outfile)
        else:
            gt.graph_draw(self.network,nodesfirst=False,
                          pos=self.pos,
                          vertex_halo=self.hl,
                          vertex_halo_color=[.2,.2,.2,.1],
                          edge_pen_width=pen_width,
                          edge_end_marker=self.e_marker,
                          vorder=self.deg,
                          edge_marker_size=10,
                          vertex_color=self.varclass,#[.5,.5,.5,.3],
                          edge_color=self.ecol,#[.5,.5,.5,.5],
                          vertex_pen_width=1.5,
                          vertex_size=self.od,
                          vertex_text=self.nm,
                          vcmap=(cmap,alpha),
                          edge_control_points=self.control,
                          vertex_fill_color=self.varclass,#deg,
                          vertex_font_size=self.ods,
                          vertex_text_color=[.1,.1,.1,.8],
                          vertex_text_position=text_pos,
                          output=self.outfile)
             
