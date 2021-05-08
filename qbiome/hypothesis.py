import pygraphviz as pgv
import networkx as nx
import numpy as np
import re
import sys
import os
from scipy import stats
import warnings
import pandas as pd
import glob
from tqdm import tqdm
warnings.filterwarnings('ignore')

class Hypothesis(object):
    """Generate and analyze hypotheses from models inferred.  Assume biomes_timestamp format is biome_timestamp.
       Also assume that dotname or decision tree name format is biome_timestamp.dot

       Mathematical model of causal hypothesis:

       ---

       ```
       Local Marginal Regulation Coefficient: Aiming to estimate the up-regulatory/down-regulatory influence of a source organism/entity on a target organism/entity, where regulation effects are causally localized in time (future cannot affect the past) with limited memory, and potential confounding effects from other entities/organisms are marginalized out.
       ```

       Let us assume a general dependency between stochastic processes \(\\nu,u, \omega \) :

       $$ \\nu_t = \\phi(u_{\leftarrow t},\omega_{\leftarrow t}) $$

      We estimate the sign of \( \\alpha_t\) in a locally linear marginalized relationship \( \\nu_t = \\alpha_t u_{t'} + c \) with \(t' \in [ t-\delta, t] \) as follows:

    Attributes:
       qnet_orchestrator (qbiome.QnetOrchestrator): instance of qbiome.QnetOrchestrator with trained qnet model
       model_path (str, optional): ath to directory containing generated decision trees in dot format (Default value = None)
       no_self_loops (bool, optional): If True do not report self-loops in hypotheses  (Default value = True)
       causal_constraint (float, optional): lag of source inputs from target effects. >= 0 is causal  (Default value = 0)
       total_samples (int, optional): total number of samples used to construct decision model  (Default value = 100)
       detailed_labels (bool, optional): if True, decision tree models have detailed output  (Default value = False)
       MAPNAME (str): path to dequantization map

    """

####    def __init__(self,qnet_orchestrator,
#    model_path=None,
#    no_self_loops=True,
#    causal_constraint=0,
#    total_samples=100,
#    detailed_labels=False):

    def __init__(self,
                 qnet_orchestrator=None,
                 quantizer=None,
                 quantizer_mapfile=None,
                 model_path=None,
                 no_self_loops=True,
                 causal_constraint=0,
                 total_samples=100,
                 detailed_labels=False):
        """

        """

        self.time_start = None
        self.time_end = None

        self.model_path = model_path
        self.qnet_orchestrator = qnet_orchestrator
        self.quantizer = quantizer


        if all(v is None for v in[qnet_orchestrator,quantizer,quantizer_mapfile]):
            raise Exception('Either qnet_orchestrator or quantizer or quantizer_mapfile must be provided to Hypothesis')

        if self.qnet_orchestrator is not None:
            self.quantizer = self.qnet_orchestrator.quantizer
        if self.quantizer is not None:
            self.variable_bin_map = self.quantizer.variable_bin_map
        else:
            self.variable_bin_map = np.load(quantizer_mapfile, allow_pickle=True)

        self.biomes_timestamp = [x for x in self.variable_bin_map.keys()]

        self.biomes = list(set(['_'.join(x.split('_')[:-1])
                                for x in self.biomes_timestamp]))

        self.NMAP = self.variable_bin_map

        if self.quantizer is not None:
            self.LABELS = quantizer.labels
        else:
            warnings.warn("Using manually coded labels. Provide quantizer to Hypothesis")
            self.LABELS = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}

        self.total_samples = total_samples
        self.detailed_labels = detailed_labels

        self.decision_tree = None
        self.tree_labels = None
        self.tree_edgelabels = None
        self.TGT = None
        self.SRC = None
        self.no_self_loops = no_self_loops

        self.causal_constraint = causal_constraint
        self.hypotheses=pd.DataFrame(columns=['src','tgt','time_tgt','lomar','pvalue'])


    def src_time_constraint(self,source,target):
        """Constrain which time points for source can be
           considered as inputs when computing source to target influences.
           If causal_constraint is None, there are no restrictions.
           Otherwise, only sources that are at least causal_constraint
           units of time prior to the target may be considered. Default is 0.
           Negative values are possible.

        Args:
          source (str): full source name in biome_timestamp format
          target (str): full target name in biome_timestamp format

        Returns:
          bool: Truth value of src acceptance

        """
        _, source_biome_full, source_timestamp, _ \
            = re.split(r'(.*)_(\d+)', source)
        _, target_biome_full, target_timestamp, _ \
            = re.split(r'(.*)_(\d+)', target)

        if self.causal_constraint is not None:
            if (float(source_timestamp)
            + self.causal_constraint
                <= float(target_timestamp)):
                return True
            return False
        return True


    def deQuantize_lowlevel(self,
                    letter,
                    bin_arr):
        """Low level dequantizer function

        Args:
          letter (str): quantized level (str) or nan
          bin_arr (numpy.ndarray): 1D array of floats, which are quantization levels from trained quantizer.

        Returns:
          float: dequantized value

        """

        if letter is np.nan or letter == 'nan':
            return np.nan
        lo = self.LABELS[letter]
        hi = lo + 1
        val = (bin_arr[lo] + bin_arr[hi]) / 2
        return val


    def deQuantizer(self,
                    letter,
                    biome_prefix,
                    timestamp_list=None,
                    time_start=None,
                    time_end=None):
        """Dequantizer function calling low level deQuantize_lowlevel to account
           for the possibility of multiple timestamps being averaged over,
           and ability to operate with incomplete biome names.

        Args:
          letter (str): quantized level (str) or nan
          biome_prefix (str): prefix of biome name
          timestamp_list (numpy.ndarray, optional): 1D array of int time-stamps to consider. (Default value = None)
          time_start (int, optional): Start time. (Default value = None)
          time_end (int, optional): End time. (Default value = None)

        Returns:
          float: median of dequantized value

        """
        vals = []
        if timestamp_list is None:
            if time_start is None and self.time_end is None:
                # average over all
                for biome_key in self.NMAP:
                    if biome_prefix in biome_key:
                        vals.append(
                            self.deQuantize_lowlevel(letter,
                                             self.NMAP[biome_key]))
            elif time_start is None:
                time_end = int(time_end)
                for biome_key in self.NMAP:
                    _, biome_full, time, _ \
                        = re.split(r'(.*)_(\d+)', biome_key)
                    time = int(time)
                    if (time <= time_end) and (biome_prefix in biome_key):
                        vals.append(
                            self.deQuantize_lowlevel(letter, self.NMAP[biome_key]))
            elif time_end is None:
                time_start = int(time_start)
                for biome_key in self.NMAP:
                    _, biome_full, time, _ = re.split(r'(.*)_(\d+)', biome_key)
                    time = int(time)
                    if (time_start <= time) and (biome_prefix in biome_key):
                        vals.append(
                            self.deQuantize_lowlevel(letter, self.NMAP[biome_key]))
            else: # both present
                time_start = int(time_start)
                time_end = int(time_end)
                for biome_key in self.NMAP:
                    _, biome_full, time, _ = re.split(r'(.*)_(\d+)', biome_key)
                    time = int(time)
                    if (time_start <= time <= time_end) and (
                            biome_prefix in biome_key):
                        vals.append(
                            self.deQuantize_lowlevel(letter,
                                             self.NMAP[biome_key]))
        else:
            for biome_key in self.NMAP:
                _, biome_full, time, _ = re.split(r'(.*)_(\d+)', biome_key)
                time = int(time)
                if time in timestamp_list:
                    vals.append(self.deQuantize_lowlevel(letter, self.NMAP[biome_key]))

        return np.median(vals)


    def getNumeric_internal(self,
               dict_id_reached_by_edgelabel,
               bin_name,
               timestamp_list=None,
               t0=None,
               t1=None):
        """Dequantize labels on graph non-leaf nodes

        Args:
          dict_id_reached_by_edgelabel (dict[int,list[str]]): dict mapping nodeid to array of letters with str type
          bin_name (str): biome name in biome_timestamp format
          timestamp_list (numpy.ndarray, optional): 1D array of int Time stamps to consider (Default value = None)
          t0 (int, optional): Start time (Default value = None)
          t1 (int, optional): End time (Default value = None)

        Returns:
          dict[int,float]: dict mapping nodeid to  dequantized values of float type

        """

        biome_prefix = '_'.join(bin_name.split('_')[:-1])

        if timestamp_list is None:
            if (t0 is None) and (t1 is None):
                t0 = int(bin_name.split('_')[-1])-1
                t1 = int(bin_name.split('_')[-1])+1
        R={}
        for k in dict_id_reached_by_edgelabel:
            v = dict_id_reached_by_edgelabel[k]
            R[k]=np.median(
                np.array([self.deQuantizer(
                    str(x).strip(),
                    biome_prefix,
                    timestamp_list=timestamp_list,
                    time_start=t0,
                    time_end=t1) for x in v]))
        return R


    def getNumeric_at_leaf(self,
                    Probability_distribution_dict,
                    Sample_fraction,
                    timestamp_list=None,
                    t0=None,
                    t1=None):
        """Dequantize labels on graph leaf nodes to return mean and sample standard deviation of outputs

        Args:
          Probability_distribution_dict (dict[int, numpy.ndarray[float]]): dict mapping nodeid to probability distribution over output labels at that leaf node
          Sample_fraction (dict[int,float]): dict mapping nodeids to sample fraction captured by that leaf node
          timestamp_list (numpy.ndarray[int], optional): 1D array of int. Time stamps to consider (Default value = None)
          t0 (int, optional): start time (Default value = None)
          t1 (int, optional): end time (Default value = None)

        Returns:
          float,float: mean and sample standard deviation

        """
        bin_name=self.TGT
        biome_prefix = '_'.join(bin_name.split('_')[:-1])

        if timestamp_list is None:
            if (t0 is None) and (t1 is None):
                t0 = int(bin_name.split('_')[-1])-1
                t1 = int(bin_name.split('_')[-1])+1

        # ----------------------------------------
        # Q is 1D array of dequantized values
        # corresponding to levels for TGT
        # ----------------------------------------
        Q=np.array([self.deQuantizer(
            str(x).strip(),
            biome_prefix,
            timestamp_list=timestamp_list,
            time_start=t0,
            time_end=t1)
                    for x in self.LABELS.keys()]).reshape(
                            len(self.LABELS),1)

        mux=0
        varx=0
        for k in Probability_distribution_dict:
            p = Probability_distribution_dict[k]

            mu_k=np.dot(p.transpose(),Q)
            var_k=np.dot(p.transpose(),(Q*Q))-mu_k*mu_k

            mux = mux + Sample_fraction[k]*mu_k
            varx = varx + Sample_fraction[k]*var_k
        return mux,np.sqrt(varx/self.total_samples)


    def regularize_distribution(self,prob,l,e=0.005):
        """Regularize probability distribution
           using exponential decay to map non-detailed output of a
           single maximum likelihood label to a probability distribution.
           Used when detailed output is not available.

        Args:
          prob (float): probability of single output label of type str
          e (float, optional): small value to regularize return probability of 1.0 (Default value = 0.005)
          l (str): output label

        Returns:
          numpy.ndarray: probability distribution

        """
        labels=np.array(list(self.LABELS.keys()))
        yy=np.ones(len(labels))*((1-prob-e)/(len(labels)-1))
        yy[np.where(labels==l)[0][0]]=prob-e
        dy=pd.DataFrame(yy).ewm(alpha=.8).mean()
        dy=dy/dy.sum()
        return dy.values


    def leaf_output_on_subgraph(self,nodeset):
        """Find the mean and sample standard deviation of output
           in leafnodes reachable from nodeset, along with fraction of samples
           captures by this subgraph

        Args:
          nodeset (numpy.ndarray): 1D array of nodeids

        Returns:
          tuple(float,float), float: mean, sample standard deviation and sample fraction

        """

        ## cLeaf is the set of leaf nodes reachable from nodeset
        # oLabels is the output labels for target and
        # is a dict mapping leafnode id to output label letter
        #
        # frac and prob are sample fraction and probability of
        # output label in leaf node, parsed from dotfile
        #
        # SUM is the total sample fraction captured by nodeset
        cLeaf=[x for x in nodeset
               if self.decision_tree.out_degree(x)==0
               and self.decision_tree.in_degree(x)==1]
        oLabels={k:str(v.split('\n')[0])
                 for (k,v) in self.tree_labels.items()
                 if k in cLeaf}

        frac={k:float(v.split('\n')[2].replace('Frac:',''))
              for (k,v) in self.tree_labels.items()
              if k in cLeaf}
        if not self.detailed_labels:
            prob={k:float(v.split('\n')[1].replace('Prob:',''))
                  for (k,v) in self.tree_labels.items()
                  if k in cLeaf}

            ## Get a kernel based distribution here.
            # self.alphabet=['A',...,'E']
            # prob is regularize_distributioned to get a dict {nodeid: [p1,..,pm]}
            prob__={k:self.regularize_distribution(prob[k],oLabels[k])
                    for k in prob}
            prob=prob__
        else:
            prob={k:self.get_vector_from_dict(v.split('\n')[1].replace('Prob:',''))
                  for (k,v) in self.tree_labels.items()
                  if k in cLeaf}

        SUM=np.array(frac.values()).sum()

        ## mean and sample estimate of standard deviation
        mu_X,sigma_X=self.getNumeric_at_leaf(prob,frac)
        return (mu_X,sigma_X),SUM


    def getHypothesisSlice(self,nid):
        """Generating impact of node nid with source label prefix. Note that there can be multiple
           nodes in the tree with label that match with the source label prefix.

        Args:
          nid (int): nodeid

        Returns:
          [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html): dataframe of hypotheses fragment with xvalue, ymean and y std dev

        """

        cNodes=list(nx.descendants(
            self.decision_tree,nid))
        nextNodes=nx.neighbors(
            self.decision_tree,nid)

        nextedge={}
        edgeProp={}
        SUM=0.
        for nn in list(nextNodes):
            nextedge[nn]=[str(x) for x in
                          self.tree_edgelabels[(
                              nid,nn)].split('\\n')]
            if len(list(nx.descendants(
                    self.decision_tree,nn))) == 0:
                res,s=self.leaf_output_on_subgraph([nn])
            else:
                res,s=self.leaf_output_on_subgraph(list(
                    nx.descendants(self.decision_tree,nn)))
            edgeProp[nn]=res
            SUM=SUM+list(s)[0]

        # nextedge is dict: {nodeid nn: letter array by which child nn is reached}
        num_nextedge=self.getNumeric_internal(
            nextedge,
            bin_name
            =self.tree_labels[str(
                nid)])
        for (k,v) in edgeProp.items():
            num_nextedge[k]=np.append(
                num_nextedge[k],[v[0],v[1]])

        RF=pd.DataFrame(num_nextedge)
        RF.index=['x', 'y','sigmay']
        return RF


    def createTargetList(self,
                      source,
                      target,
                      time_end=None,
                      time_start=None):
        """Create list of decision trees available within time points in model_path

        Args:
          source (str): source name in biome_timestamp format
          target (str): target name in biome_timestamp format
          time_end (int, optional): start time (Default value = None)
          time_start (int, optional): end time (Default value = None)

        Returns:
          list[str]: list of paths to decision tree models in qnet

        """
        self.TGT = target
        self.SRC = source

        if self.model_path is not None:
            decision_trees = glob.glob(
                os.path.join(self.model_path,
                             target)+'*.dot')
            decision_trees_ = [x for x in decision_trees
                               if (time_start
                                   <= float(re.split(
                                       r'(.*)_(\d+)',x)[-2])
                                   <= time_end)]
        else:
            raise Exception('self.model_path is not set')
        return decision_trees_


    def get_lowlevel(self,
            source,
            target,
            time_end=None,
            time_start=None):
        """Low level evaluation call to estimate local marginal regulation  \( \\alpha \)

        Args:
          source (str): source
          target (str): target
          time_end (int, optional): end time (Default value = None)
          time_start (int, optional): start time (Default value = None)

        Returns:

        """
        self.TGT = target
        self.SRC = source

        decision_trees = self.createTargetList(
            source,
            target,
            time_end,
            time_start)
        grad_=[]
        # can we do this in parallel
        for tree in decision_trees:
            self.TGT = os.path.basename(tree).replace('.dot','')
            gv = pgv.AGraph(tree,
                            strict=False,
                            directed=True)

            self.decision_tree = nx.DiGraph(gv)
            self.time_start = time_start
            self.time_end = time_end

            self.tree_labels = nx.get_node_attributes(
                self.decision_tree,'label')
            self.tree_edgelabels = nx.get_edge_attributes(
                self.decision_tree,"label")

            nodes_with_src=[]
            for (k,v) in self.tree_labels.items():
                if self.SRC in v:
                    if self.src_time_constraint(v,self.TGT):
                        nodes_with_src=nodes_with_src+[k]

            if len(nodes_with_src)==0:
                continue

            RES=pd.concat([self.getHypothesisSlice(i).transpose()
                           for i in nodes_with_src])

            grad,pvalue=self.getAlpha(RES)
            #RES.to_csv('tmp.csv')
            #if RES.index.size > 2:
            #    quit()

            #grad=stats.linregress(
            #    RES.x_.values,
            #    RES.muy.values).slope

            if np.isnan(grad):
                warnings.warn(
                    "Nan encountered in causal inferrence")
                grad=np.median(
                    RES.y.values)/np.median(
                        RES.x.values)

            ns_ = re.split(r'(.*)_(\d+)', self.TGT)
            self.hypotheses = self.hypotheses.append(
                {'src':self.SRC,
                 'tgt':''.join(ns_[:-2]),
                 'time_tgt':float(ns_[-2]),
                 'lomar':float(grad),
                 'pvalue':pvalue},
                ignore_index = True)
        return


    def get(self,
            source=None,
            target=None,
            time_end=None,
            time_start=None):
        """Calculate local marginal regulation  \( \\alpha \). When source or target is not specified, we calculate for all entities available on model path. Populates self.hypotheses.

        Args:
          source (str, optional): source (Default value = None)
          target (str, optional): target (Default value = None)
          time_end (int, optional): end time (Default value = None)
          time_start (int optional): start time (Default value = None)

        Returns:

        """

        if source is None:
            source = self.biomes
        else:
            if isinstance(source,str):
                source=[source]
        if target is None:
            target = self.biomes
        else:
            if isinstance(target,str):
                target=[target]

        for tgt_biome_ in tqdm(target):
            for src_biome_ in source:
                if (src_biome_ == tgt_biome_) and self.no_self_loops:
                    continue
                self.get_lowlevel(src_biome_,
                          tgt_biome_,
                          time_end=time_end,
                          time_start=time_start)

        if self.no_self_loops:
            self.hypotheses=self.hypotheses[~(self.hypotheses.src==self.hypotheses.tgt)]

        return


    def to_csv(self, *args, **kwargs):
        """Output csv of hypotheses inferred. Arguments are passed to pandas.DataFrame.to_csv()

        Args:
          *args: optional arguments to [pandas.to_csv()]( https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)
          **kwargs: optional keywords to [pandas.to_csv()]( https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)

        Returns:

        """
        self.hypotheses.to_csv(*args, **kwargs)


    def to_dot(self,filename='tmp.dot',
               hypotheses=None,
               square_mat=False):
        """Output dot file of hypotheses inferred.

        Args:
          filename (str, optional): filename of dot outpt (Default value = 'tmp.dot')
          hypotheses ([pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), optional): If provided use this instead of self.hypotheses (Default value = None)
          square_mat (bool, optional): If True resturn a heatmap matrix as filename+'sq.csv' (Default value = False)

        Returns:

        """

        if hypotheses is None:
            df=self.hypotheses.copy()
        else:
            df=hypotheses

        df=df.groupby(['src','tgt']).median().reset_index()
        df=df.pivot(index='src',columns='tgt',values='lomar')
        df=df.fillna(0)

        index = df.index.union(df.columns)
        df = df.reindex(index=index, columns=index, fill_value=0)

        df=df.sort_index()
        if square_mat:
            df.to_csv(filename.replace('.dot','sq.csv'))

        G = nx.from_pandas_adjacency(df,create_using=nx.DiGraph())

        from networkx.drawing.nx_agraph import write_dot
        write_dot(G,filename)

        return


    def getAlpha(self,dataframe_x_y_sigmay,N=500):
        """Carryout regression to estimate   \( \\alpha \). Given mean and variance of each y observation, we
           increase the number of pints by drawing N samples from a normal distribution of mean y and std dev sigma_y.
           The slope and p-value of a linear regression fit is returned

        Args:
          dataframe_x_y_sigmax (pandas DataFrame): columns x,y,sigmay
          N (int): number of samples drawn for each x to set up regression problem

        Returns:
          float,float: slope and p-value of fit

        """
        gf=pd.DataFrame(np.random.normal
                        (dataframe_x_y_sigmay.y,
                         dataframe_x_y_sigmay.sigmay,
                         [N,dataframe_x_y_sigmay.index.size]))

        RES=[dataframe_x_y_sigmay[['y','x']]]
        for i in gf.columns:
            xf=pd.DataFrame(gf.iloc[:,i])
            xf.columns=['y']
            xf['x']=dataframe_x_y_sigmay.x[i]
            RES.append(xf)
        RES=pd.concat(RES).dropna()

        lr=stats.linregress(RES.x,RES.y)
        return lr.slope,lr.pvalue


    def get_vector_from_dict(self,str_alph_val):
        """Calculate a probability distribution from string representation of
           alphabet : value read from decision tree models

        """
        vec_alph_val=str_alph_val.split()

        dict_label_float={}

        for x in vec_alph_val:
            y=x.split(':')
            dict_label_float[y[0]]=float(y[1])

        prob_dist = np.zeros(len(self.LABELS.keys()))
        for i in dict_label_float:
            prob_dist[self.LABELS[i]] = dict_label_float[i]


        return prob_dist/prob_dist.sum()


    def trim_hypothesis(self,alternate_hypothesis_dataframe):
        """Compate current hypothesis dataframe with alternate_hypothesis_dataframe

        Args:
          alternate_hypothesis_dataframe ([pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)): alternate dataframe

        Returns:
          [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html): manipulated dataframe

        """
        df=self.hypotheses.copy()

        #df.set_index(['src','tgt']).merge


        return df
