from plot_functions import logPlot
import pandas as pd

#%% [markdown]
# ## Applying the classification model to new data
# 
# Now that we have a trained facies classification model we can use it to identify facies in wells that do not have 
# core data.  In this case, we will apply the classifier to two wells, but we could use it on any number of wells for 
# which we have the same set of well logs for input.
# 
# This dataset is similar to the training data except it does not have facies labels.  It is loaded into a dataframe 
# called `test_data`.
#%%

def newTest(scaler, clf, facies_colors):
    well_data = pd.read_csv('validation_data_nofacies.csv')
    well_data['Well Name'] = well_data['Well Name'].astype('category')
    well_features = well_data.drop(['Formation', 'Well Name', 'Depth'], axis=1)

    #%% [markdown]
    # The data needs to be scaled using the same constants we used for the training data.
    #%%
    X_unknown = scaler.transform(well_features)

    #%% [markdown]
    # Finally we predict facies labels for the unknown data, and store the results in a `Facies` column of the `test_data` 
    # dataframe.
    #%%
    #predict facies of unclassified data
    y_unknown = clf.predict(X_unknown)
    well_data['Facies'] = y_unknown
    well_data

    well_data['Well Name'].unique()

    #%% [markdown]
    # We can use the well log plot to view the classification results along with the well logs.
    #%%
    logPlot(well_data[well_data['Well Name'] == 'STUART'], facies_colors=facies_colors)
    logPlot(well_data[well_data['Well Name'] == 'CRAWFORD'],facies_colors=facies_colors)
