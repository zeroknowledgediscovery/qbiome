import statsmodels.api as sm
lowess = sm.nonparametric.lowess
import warnings
import pylab as plt
import pandas as pd

def saveFIG(filename='tmp.pdf',
            axis=False,
            transparent=True):
    """save fig for publication

    Args:
      filename (str, optional): filename to save figure. (Default value = 'tmp.pdf')
      axis (bool, optional): if True then show axis. (Default value = False)
      transparent (bool, optional): if True background is transparent. (Default value = True)

    Returns:

    """
    import pylab as plt
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    if not axis:
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filename,dpi=300, bbox_inches = 'tight',
                pad_inches =.1,transparent=transparent) 
    return


def qplot(df,
          preindex,
          index,
          columns,
          timeunit=None,
          var=None,
          interpolate=False,
          alpha=.9,
          lowess_fraction=0.6,
          normalize=True,
          ax=None,
          xlim0=None,
          xlim1=None,
          legend_label=None,
          filename=None,
          transparent=True,
          save=True,
          fontsize=18,
          title=None):
    """plot dataframes after pivoting and slicing

    Args:
      df (pandas.DataFrame): dataframe in long format
      preindex (str): if not None, set index to preindex
      index (str): pivot index
      columns (str): pivot columns
      timeunit (str, optional): label for unit of time. If None, set to index (Default value = None)
      var (list[str], optional): list of variables to plot (Default value = None)
      interpolate (bool, optional): remove Nans by spline fit (Default value = False)
      alpha (float, optional): parameter passed to exponential smoothing (Default value = .9)
      lowess_fraction (float): smoothing coefficient for LOWESS (Default value = 0.6)
      normalize (bool): if True normalize (Default value = True)
      ax (axis handle): If None generate figure and axes (Default value = None)
      xlim0 (float, optional): left limit of x axis (Default value = None)
      xlim1 (float, optional): right limit of x axis (Default value = None)
      legend_label (str, optional): optional suffix added to  legend (Default value = None)
      filename (str, optional): output filename including extension (Default value = None)
      transparent (bool, optional): if True background is transparent (Default value = True)
      save (bool, optional): if True save file  (Default value = True)
      fontsize (int, optional): fontsize  (Default value = 18)
      title (str, optional): title string (Default value = None)

    Returns:
      pandas.DataFrame: concatenated dataframe plotted

    """

    if timeunit is None:
        timeunit=index
        
    df=df.set_index(preindex).pivot(index=index,columns=columns)
    df.columns=[x[1] for x in df.columns]
    if interpolate:
        df=df.interpolate(method='spline',order=2,limit_direction='both')
    if ax is None:
        fig=plt.figure(figsize=[8,4])
        ax=plt.gca()
    if var is not None:
        if not isinstance(var, list):
            warning('var needs to be a list')
            var=[var]
        df_=df[var]
        biomes=var
    else:
        df_=df.copy()
        biomes=df.columns

    DF=None
    for i in biomes:
        df__=df_[i].ewm(alpha=alpha).mean()        
        w = lowess(df__.values,df__.index.values, frac=lowess_fraction)
        df__=pd.DataFrame(w,columns=[timeunit,i]).set_index(timeunit)
        if normalize:
            df__=(df__-df__.min())/(df__.max()-df__.min())
        df__.plot(ax=ax,label=i,style='-',lw=4,ms=8,alpha=.75)
        if DF is None:
            DF=df__.reset_index()
        else:
            DF=DF.merge(df__.reset_index(),on=timeunit)
        
    if legend_label is not None:
        biomes=[x+legend_label for x in biomes]
    ax.legend(biomes,loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim(xlim0,xlim1)
        
    if normalize:
        ax.set_ylim(-.1,1.1)
        tg='normalized '
    else:
        tg=''
    ax.set_ylabel(tg+'concentration',fontsize=fontsize,labelpad=10,color='.5')
    ax.set_xlabel('['+timeunit+']',fontsize=fontsize,labelpad=10,color='.5')
    if title is not None:
        ax.set_title(title,y=1.03,fontsize=fontsize+2,fontweight='demi')

    ax.tick_params(axis='x', labelsize=fontsize,labelcolor='.5' )
    ax.tick_params(axis='y', labelsize=fontsize,labelcolor='.5')
    
    if filename is None:
        filename='_'.join(biomes)+'.png'
    if save:
        saveFIG(filename,axis=True,transparent=transparent)
        
    return DF.set_index(timeunit)


def customDataFormatter(datafile,
                        metafile,
                        META_PROP,
                        COL_SELECT,
                        BIOMES=None,
                        sample_id_col_names=['Samples','sequence_barcode']):
    """custom data formatter. See example.

    Args:
      datafile (str): path to datafile 
      metafile (str): path to metadatafile
      META_PROP (list[str]): list of meta properties in metaddata file
      COL_SELECT (dict[str,str]): dict to rename columns for sample_id, subject_id, timeunit
      BIOMES (list[str]):  list of biomes of interest (Default value = None)
      sample_id_col_names:  (Default value = ['Samples','sequence_barcode']): 

    Returns:
      pandas.DataFrame,pandas.DataFrame: data and subject_id vs meta properties
    """ 
    
    df_=pd.read_csv(datafile).set_index(sample_id_col_names[0])
    mf=pd.read_csv(metafile).set_index(sample_id_col_names[1])
    mf.index.name='Samples'
    mf=mf.reset_index()
    df=df_.merge(mf,on='Samples')
    df=df.rename(columns=COL_SELECT)
    if BIOMES is None:
        BIOMES=list(df_.columns.values[1:])
    df1=df[['sample_id','subject_id','week']+BIOMES].drop(['subject_id','week'],axis=1)
    df2=df[['sample_id','subject_id','week']]
    data=df1.melt(id_vars='sample_id',value_vars=BIOMES).merge(df2,on='sample_id')
    property_map=df[['subject_id']+META_PROP]
    
    return data,property_map
