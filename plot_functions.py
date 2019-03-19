import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np

def getMinMax(logs):
    min = logs.min()
    if (np.isnan(min)):
        min = 0
    max = logs.max()
    if (np.isnan(max)):
        max = 100
    return min,max

def logPlot(logs, seam_labels, seam_colors):
    if (len(logs) == 0):
        return

    #make sure logs are sorted by depth
    logs = logs.sort_values(by='FROM')
    cmap_facies = colors.ListedColormap(
            seam_colors[0:len(seam_colors)], 'indexed')
    
    ztop=logs.Z.min()
    zbot=logs.Z.max()
    
    exp = np.expand_dims(logs['SEAM'].values,1)
    cluster=np.repeat(exp, 200, 1)
    
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(4, 12))
    rashMask = np.isfinite(logs.RASH)
    ax[0].plot(logs.RASH[rashMask], logs.Z[rashMask], '-g')
    rfsiMask = np.isfinite(logs.RFSI)
    ax[1].plot(logs.RFSI[rfsiMask], logs.Z[rfsiMask], '-')
    im=ax[2].imshow(cluster,interpolation='none',aspect='auto',cmap=cmap_facies)
    
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((2*' ').join(seam_labels))
    cbar.set_ticks(range(0,1))
    cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)   
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    ax[0].set_xlabel("RASH")
    min,max = getMinMax(logs.RASH)
    ax[0].set_xlim(min,max)
    
    ax[1].set_xlabel("RFSI")
    min,max = getMinMax(logs.RFSI)
    ax[1].set_xlim(min,max)
    
    ax[2].set_xlabel('Seams')
    
    ax[1].set_yticklabels([])
    ax[2].set_yticklabels([])
    ax[2].set_xticklabels([])
    f.suptitle('DHID: %s'%logs.iloc[0]['DHID'], fontsize=14,y=0.94)


def crossPlots(training_data, facies_labels, facies_color_map):    
    #save plot display settings to change back to when done plotting with seaborn
    inline_rc = dict(mpl.rcParams)

    sns.set()
    sns.pairplot(training_data.drop(['Well Name','Facies','Formation','Depth','NM_M','RELPOS'],axis=1), 
        hue='FaciesLabels', palette=facies_color_map,
        hue_order=list(reversed(facies_labels)))

    #switch back to default matplotlib plot style
    mpl.rcParams.update(inline_rc)




def compare_facies_plot(logs, compadre, seam_colors, seam_labels):
    if (len(logs) == 0):
        return

    #make sure logs are sorted by depth
    logs = logs.sort_values(by='FROM')
    cmap_facies = colors.ListedColormap(
            seam_colors[0:len(seam_colors)], 'indexed')
    
    ztop=logs.Z.min()
    zbot=logs.Z.max()
    
    exp = np.expand_dims(logs['SEAM'].values,1)
    cluster1=np.repeat(exp, 200, 1)

    exp = np.expand_dims(logs['Prediction'].values,1)
    cluster2=np.repeat(exp, 200, 1)

    f, ax = plt.subplots(nrows=1, ncols=4, figsize=(4, 12))

    rashMask = np.isfinite(logs.RASH)
    ax[0].plot(logs.RASH[rashMask], logs.Z[rashMask], '-g')
    
    rfsiMask = np.isfinite(logs.RFSI)
    ax[1].plot(logs.RFSI[rfsiMask], logs.Z[rfsiMask], '-')

    im1=ax[2].imshow(cluster1,interpolation='none',aspect='auto',cmap=cmap_facies)
    im2=ax[3].imshow(cluster2,interpolation='none',aspect='auto',cmap=cmap_facies)
    
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar=plt.colorbar(im2, cax=cax)
    cbar.set_label((2*' ').join(seam_labels))
    cbar.set_ticks(range(0,1))
    cbar.set_ticklabels('')
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)   
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)
    
    min,max = getMinMax(logs.RASH)
    ax[0].set_xlabel("RASH")
    ax[0].set_xlim(min,max)
    
    min,max = getMinMax(logs.RFSI)
    ax[1].set_xlabel("RFSI")
    ax[1].set_xlim(min,max)
    ax[1].set_yticklabels([])
    
    ax[2].set_xlabel('Seams')
    ax[2].set_yticklabels([])
    ax[2].set_xticklabels([])

    ax[3].set_xlabel('Prediction')
    ax[3].set_yticklabels([])
    ax[3].set_xticklabels([])
    
    f.suptitle('DHID: %s'%logs.iloc[0]['DHID'], fontsize=14,y=0.94)