from plot_functions import logPlot
from plot_functions import crossPlots
import matplotlib.pyplot as plt



def showLogPlotByDHID(training_data, seam_labels, seam_colors, dhid):
    rowIndexes = training_data['DHID'] == dhid
    logPlot(training_data[rowIndexes], seam_labels, seam_colors)
    plt.show(block=True)

def showCountHistogram(training_data, facies_labels, facies_colors, column, title):
    # count the number of unique entries for each facies, sort them by facies number (instead of by number of entries)
    facies_counts = training_data[column].value_counts().sort_index()
    # use facies labels to index each count
    facies_counts.index = facies_labels

    facies_counts.plot(kind='bar', color=facies_colors, title=title)
    plt.show(block=True)


def exploreData(trainingData, seamLabels, seamColors):
    # log plot of a single well
    showLogPlotByDHID(trainingData, seamLabels, seamColors, dhid='D424')
    showLogPlotByDHID(trainingData, seamLabels, seamColors, dhid='D448')
    showLogPlotByDHID(trainingData, seamLabels, seamColors, dhid='D431')
    showLogPlotByDHID(trainingData, seamLabels, seamColors, dhid='D445')

    # histogram of facies labels
    # This shows the distribution of examples by facies for the 3232 training examples in the training set.  
    # Dolomite (facies 7) has the fewest with 141 examples.  There are also only 185 bafflestone examples.  
    # Depending on the performance of the classifier we are going to train, we may consider getting more examples of these 
    # facies.
    showCountHistogram(trainingData, seamLabels, seamColors, column='ASEAM', title='Seam Count Histogram')    
     
    # Crossplots are a familiar tool in the geosciences to visualize how two properties vary with rock type.  
    # This dataset contains 5 log variables, and scatter matrix can help to quickly visualize the variation between the 
    # all the variables in the dataset.  We can employ the very useful 
    # [Seaborn library](https://stanford.edu/~mwaskom/software/seaborn/) to quickly create a nice looking scatter matrix. 
    # Each pane in the plot shows the relationship between two of the variables on the x and y axis, with each point is 
    # colored according to its facies.  The same colormap is used to represent the 9 facies.  

    #crossPlots(training_data, data.facies_labels, data.facies_color_map)
    #plt.show()
