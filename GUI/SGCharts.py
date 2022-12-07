import matplotlib as mpl, matplotlib.pyplot as plt
import pandas as pd, numpy as np
import scipy, math, datetime
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

import statsmodels.formula.api as smf, statsmodels.api as sm

default_dark = False

jupyterlab_theme = 'monokai'

pd.plotting.register_matplotlib_converters()

jupyterlab_backgrounds = {'dark':'##111111','monkai':'#272822'}
chosen_theme = 'monkau'

colors = {'nbf_red':'#E41C23','nbf_blue':'#00324D','nbf_red2':'#FFCCCB','nbf_blue2':'#BBDDE6','_fg':'black','_dark_bg':jupyterlab_backgrounds[chosen_theme] if chosen_theme in jupyterlab_backgrounds else jupyterlab_backgrounds['dark']}
backup_colors = ['xkcd:seeweed','xkcd:lilac','xkcd:pine']

def cl(i=None):
    cl2 = [colors[c] for c in colors if c[0]!='_']+backup_colors
    if i is None:
        return cl2
    if i>=len(cl2):
        return f.rand_color()
    else:
        return cl2[i]

    
dist_clrs = [(0.10793530774748605, 0.9965125450102401, 0.07573392107253074),
 (1.0, 0.0, 1.0),
 (0.0, 0.5, 1.0),
 (1.0, 0.5, 0.0),
 (0.5, 0.25, 0.5),
 (0.48611977360582104, 0.7872183348522868, 0.6373394935076268),
 (0.753655639754272, 0.9805579168996887, 0.05906639503682487),
 (0.9771830592922839, 0.010425851106580564, 0.1991737956087859),
 (0.017349425463487056, 0.48967058672267927, 0.15338898115032762),
 (0.0, 0.0, 1.0),
 (0.0, 1.0, 1.0),
 (1.0, 0.5, 1.0),
 (0.0, 0.0, 0.5),
 (0.47580595431408224, 0.255182416406355, 0.993305463324793),
 (0.9462811524159542, 0.743395937761401, 0.46963071902404796),
 (0.5149217729086369, 0.3964545637603759, 0.0009469129751373817),
 (0.011037174490858837, 0.7943014289450293, 0.43720036949487817),
 (0.5, 0.0, 0.0)]
    
def _label_number(t):
    exps = {0:'',1e6:'Mln',1e9:'Bln',1e12:'T'}
    max_exp = max([k for k in exps if k<=t])
    max_label = exps[max_exp]
    
    if max_exp>1:
        return '{:,.0f}{}'.format(t/max_exp,max_label)
    else:
        return '{:,.0f}'.format(t)
kw_scatter = {'s':2, 'color':colors['nbf_red'],'linewidth':0}

def mpl_finance_style():
    mc = mpf.make_marketcolors(up=colors['nbf_blue'],down=colors['nbf_red'],ohlc='i')
    s = mpf.make_mpf_style(marketcolors=mc)
    return s

def Cx(params={}):
    return Chart(*params).ax

class Chart():
    
    def __init__(self,name=None,figsize=None,tufte=True,custom_axes=None,ncols=1,nrows=1,
                 row_ratios=None,col_ratios=None,aspect_ratio=None,projection=None,draft=True,
                 dark=None,sharex='col',sharey='row',margin=0.3,fig=None,render=None,title_font=None,x_font=None,y_font=None,line_size=None):
    
        self.save_file = name is not None

        dark = default_dark if dark is None else dark
        stylesheet = 'classic' if not dark else 'dark_background'
        self.dark = dark
        if dark:
            colors['nbf_blue2'], colors['nbf_blue'] = '#00324D','#BBDDE6'
            colors['_fg'] = 'white'

        plt.style.use(stylesheet)
        plt.tight_layout()

        # GOLDEN RATIO AESTHETICS
        aspect_ratio = (1+np.sqrt(5))/2 if aspect_ratio is None else aspect_ratio

        # Squares should be smaller than rectangles ... (probably)
        default_figsize = 6.5 if abs(aspect_ratio) > 1.1 else 4
        named_sized = {'default':default_figsize,'small':[4,2]}
        
        if type(figsize) == str:
            figsize = named_sizes[figsize] if figsize in named_sizes else named_sizes['default']
        figsize = default_figsize if figsize is None else figsize
        if type(figsize) != type([]):
            #figsize=6.5
            #print(figsize,aspect_ratio)
            figsize = (figsize,figsize/aspect_ratio)

        self.figsize = figsize
        self.dpi = 150 if draft else 150
        self.draft = draft

        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.transparent'] = not draft

        self.sharex = None
        self.sharey = None

        #Set Font Sizes
        font = {'family' : 'Arial',
               'weight' : 'normal',
               'size': 8}
        plt.rc('font',**font)

        if tufte:
            plt.rcParams['figure.titlesize'] = 10
            plt.rcParams['figure.titleweight'] = 'bold'
            plt.rcParams['legend.fontsize'] = 8
            plt.rcParams['xtick.labelsize'] = 10
            plt.rcParams['ytick.labelsize'] = 10
            plt.rcParams['lines.linewidth'] = 1
        
        if title_font is not None:
            plt.rcParams['figure.titlesize'] = title_font
        
        if x_font is not None:
            plt.rcParams['xtick.labelsize'] = x_font
        
        if y_font is not None:
            plt.rcParams['ytick.labelsize'] = y_font
        
        if line_size is not None:
            plt.rcParams['lines.linewidth'] = line_size
            

        render = (fig is not None) if (render is None) else render

        row_ratios = np.ones(nrows) if row_ratios is None else row_ratios
        col_ratios = np.ones(ncols) if col_ratios is None else col_ratios

        custom_axes = (True if nrows*ncols > 1 else False) if custom_axes is None else custom_axes

        if custom_axes:
            # Not a fan of grid aspects
            self.fig = plt.figure() if fig is None else fig

            self.grid = self.fig.add_gridspec(ncols=ncols,nrows=nrows,height_ratios=row_ratios,width_ratios=col_ratios)
            plt.subplots_adjust(hspace=margin,wspace=margin)

            try:
                self.axes = [self.fig.add_subplot(subplot_spec) for subplot_spec in self.grid]
                self.sharex = sharex
                self.sharey = sharey
                self.ax = self.axes[0]
            except:
                self.axes = None
        else:
            # Default for ez charts
            if nrows*ncols > 1:
                self.fig,self.axes = plt.subplots(nrows,ncols,sharex=sharex,sharey=sharey,gridspec_kw={'width_ratios':col_ratios,'height_ratios':row_ratios})
                plt.subplots_adjust(hspace=margin,wspace=margin)
            else:
                self.fig = plt.figure()
                self.axes = None
            if projection is not None:
                self.ax = self.fig.gca(projection=projection)
            else:
                self.ax = self.fig.gca()

        facecolor = (1,1,1,1) if not dark else colors['_dark_bg']

        plt.rcParams['axes.facecolor'] = facecolor
        self.fig.patch.set_facecolor(facecolor)
        self.tufte=tufte

        self.name = name
        self.fig.suptitle(self.name,ha='left' if tufte else 'center',x=0.1 if tufte else 0.5,y=1.0,va='bottom')

        # filename
        self.filename = self._filename(self.name,draft=draft)

        if render:
            return self.render()
        return None
    
    def _filename(self,name,draft):
        illegal_characters = {'\n':'_',' ':'-','%':'pct','#':'num'}
        splitter_chars = list(set([illegal_characters[k] for k in illegal_characters]))
        filename = name if name is not None else 'fig'
        
        try:
            filename = filename + '_' + self.ax.get_title() if self.ax.get_title() != '' else filename
        except:
            pass
        
        for c in illegal_characters:
            filename = filename.replace(c,illegal_characters[c])
        
        filename = ''.join(c for c in filename if (c.isalnum() or c in splitter_chars))
        extension = 'png' if draft else 'svg'
        
        filename = '.'.join([filename,extension])
        return filename
    
    def log_scale(self,x=True,y=True,label_x = False, label_y=False):
        if x:
            self.ax.set_xscale('log')
            if label_x:
                self.ax.set_xlabels([_label_number(t) for t in ch.ax.get_yticks()])
        if y:
            self.ax.set_yscale('log')
            if label_y:
                self.ax.set_ylabels([_label_number(t) for t in ch.ax.get_yticks()])
    
    def rotate_xticks(self,angle=90,ax=None):
        ax = self.ax if ax is None else ax
        [t.set_rotation(angle) for t in ax.get_xticklabels()]
        return None
    
    def remove_legend(self):
        ax.get_legend().remove()
        return None
    
    def pct_scale(self,x=True,y=True):
        if x:
            self.ax.set_xlim(max(0,self.ax.get_ylim()[0]),min(self.ax.get_xlim()[0],1))
        if y:
            self.ax.set_ylim(max(-1,self.ax.get_ylim()[0]),min(self.ax.get_ylim[1],1))
    
    def _twin_axes(self,ax):
        twin_axes = []
        s = ax.get_shared_x_axes().get_siblings(ax)
        if len(s)>1:
            for tx in [tx for tx in s if tx is not ax]:
                if tx.bbox.bounds == ax.bboc.bounds:
                    twin_axes.append(tx)
        return twin_axes
    
    def _get_shapes(self,ax):
        handles, labels = ax.get_legend_handles_labels()
        handles = [h for h in handles if h.get_label()[0]!='_']
        return handles
    
    # def _render() lines 244 to 342
    
    def _render(self,ax,grid_axes,spines,loc,legend_columns,reverse_legend,ticks,legend_title):
        
        if False:
            if len(self.axes)>0:
                xlim = None
                ylim = None
                for ax in self.axes:
                    if self.sharex=='col':
                        if xlim is None:
                            xlim = list(ax.get_xlim())
                        else:
                            xlim[0] = min(ax.get_xlim()[0], xlim[0])
                            xlim[1] = max(ax.get_xlim()[1], xlim[1])
                        
                    if self.sharey=='row':
                        if ylim is None:
                            ylim = list(ax.get_ylim())
                        else:
                            ylim[0] = min(ax.get_ylim()[0], ylim[0])
                            ylim[1] = max(ax.get_ylim()[1], ylim[1])
                            
                for ax in self.axes:
                    if self.sharex=='col':
                        ax.set_xlim(xlim)
                    if self.sharey=='row':
                        ax.set_ylim(ylim)
        
        #Hide Spines and Ticks
        spines = [] if spines is None else spines
        n_spines_visible = 0
        for spine in ['top','left','bottom','right']:
            ax.spines[spine].set_visible(spine in spines)
            n_spines_visible = n_spines_visible + (1 if spine in spines else 0)
            
        if (n_spines_visible < 4):
            if self.dark:
                ax.set_facecolor((1,1,1,0))
            else:
                pass
        elif not self.dark:
            pass
        
        ax.tick_params(top='top' in ticks)
        ax.tick_params(bottom='bottom' in ticks)
        ax.tick_params(left='left' in ticks)
        ax.tick_params(right='right' in ticks)
        
        shapes = self._get_shapes(ax=ax)
        for tx in self._twin_axes(ax=ax):
            shapes.extend(self._get_shapes(ax=tx))

        shapes = shapes[::1 if reverse_legend else -1]
        if len(shapes)>0:
            shape_labels = [l.get_label() for l in shapes]
            if loc is not None:
                if loc=='below':
                    pass
                else:
                    ax.legend(shapes,shape_labels,frameon=True,loc=loc,ncol=legend_columns,prop={'size':8},title=legend_title)
            else:
                try:
                    ax.get_legend().remove()
                except:
                    pass
        if (grid_axes=='none') or (grid_axes is None):
            ax.grid(False)
        else:
            ax.grid(True,axis=grid_axes,linestyle='-',linewidth=0.25,color='grey',zorder=-1)
        if (ax.get_ylabel()=='') and (len(ax.get_lines())>0):
            implied_label = ax.get_lines()[0].get_label()
            for l in [' (left)',' (right)']:
                implied_label = implied_label.replace(l,'')
            ax.set_ylabel(implied_label)
            
        try:
            if ax.get_ylabel()[0]=='_':
                ax.set_ylabel(ax.get_ylabel()[1:])
            if ax.set_xlabel()[0]=='_':
                ax.set_xlabel(ax.get_xlabel()[1:])
        except:
            pass
        
        ax.set_ylabel(ax.get_ylabel(),ha='left',y=0)
        ax.set_xlabel(ax.get_xlabel(),ha='left',x=0)
        ax.set_title(ax.get_title(),ha='left',x=0)
    
    def render(self,grid_axes='both',spines=['left','top','right','bottom'],loc='best',legend_columns=1,reverse_legend= False,ticks=None,legend_title=None,filename=None):
        if ticks is None:
            ticks = spines
            if grid_axes!='both':
                if grid_axes=='x':
                    ticks = [t for t in ticks if t not in ['left','right']]
                if grid_axes=='y':
                    ticks = [t for t in ticks if t not in ['top','bottom']]
        #ax.legend()
        if self.tufte:
            for i,ax in enumerate(self.fig.axes):
                self._render(ax,grid_axes=grid_axes,spines=spines,loc=loc,legend_columns=legend_columns, reverse_legend=reverse_legend,ticks=ticks,legend_title=legend_title)
                True
        self.filename = self._filename(self.name,draft=self.draft)
        #ax.get_legend().remove() #ADDED!
        if self.save_file:
            self.fig.savefig(self.filename,bbox_inches='tight',dpi=self.dpi,figsize=self.figsize)
        #if file_name is not None:
        return self.filename
