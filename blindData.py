from validateModel import validate
from plot_functions import compare_facies_plot

#%% [markdown]
# ## Applying the classification model to the blind data
# 
# We held a well back from the training, and stored it in a dataframe called `blind`:
#%%

def blindTest(blind, scaler, clf, facies_labels, facies_colors, adjacent_facies):
    blind

    #%% [markdown]
    # The label vector is just the `Facies` column:
    #%%
    y_blind = blind['SEAM'].values

    #%% [markdown]
    # We can form the feature matrix by dropping some of the columns and making a new dataframe:
    #%%
    well_features = blind.drop(['DHID', 'FROM', 'TO', 'ASEAM', 'SEAM'], axis=1)

    #%% [markdown]
    # Now we can transform this with the scaler we made before:
    #%%
    x_blind = scaler.transform(well_features)

    #%% [markdown]
    # Now it's a simple matter of making a prediction and storing it back in the dataframe:
    #%%
    y_pred = clf.predict(x_blind)
    blind['Prediction'] = y_pred

    #%% [markdown]
    # Let's see how we did with the confusion matrix:
    #%%
    print('Blind test data')
    validate(clf, x_blind, y_blind, facies_labels, adjacent_facies)

    compare_facies_plot(blind, 'Prediction', facies_colors, facies_labels)

