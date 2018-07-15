import pandas as pd
import os
import sys 
import random
import numpy as np

from utils import train_validation_test_split, DataManager
from nn import NeuralRegressor


HERE_PATH = os.path.dirname(__file__)
root_path = os.path.join(HERE_PATH, '..')
sys.path.append(root_path)

data_path = os.path.join(root_path, 'data//aap9112_Data_File_S1.csv')
dataframe = pd.DataFrame.from_csv(data_path)

# fields for one-hot encoding
data_fields = ['Reactant_1_Name', 'Reactant_2_Name', 'Ligand_Short_Hand',
                       'Reagent_1_Short_Hand', 'Solvent_1_Short_Hand']
# get uniqe names for each field
unique_fields_list = [list(set(dataframe[field])) for field in data_fields]


class ChemicalSpace():
    ''' Abstract class of chemical space for simulating navigation of 
        reaction space using machine learning'''
    def __init__(self, df, seed=None):
        # seed for reproduciibity
        random.seed(seed)
        self.df = df
        self.index = list(dataframe.index)
        self.explored_space_index = []
        self.space_len = len(self.index)
        # randomize the indexes
        random.shuffle(self.index)
        
        
        
    def random_guess(self, percent):
        '''Returns the dataframe with random rxns from chemical space'''
        
        num_of_rxns = int(self.space_len * percent)
        
        random_rxns_idxs = []
        
        for idx  in self.index:
            if idx not in self.explored_space_index:
                self.explored_space_index.append(idx)
                random_rxns_idxs.append(idx)
                if len(random_rxns_idxs) == num_of_rxns:
                    break
        random_df = self.df.loc[random_rxns_idxs]
        return random_df
    
    
    
    
    def get_unused(self):
        ''' Return a dataframe of all reactions which has not been explored so far'''
        
        unused_rxn_idxs = []
        
        for idx in self.index:
            if idx not in self.explored_space_index:
                unused_rxn_idxs.append(idx)
                
        unused_rxns = self.df.loc[unused_rxn_idxs]
        return unused_rxns
        
    def get_explored(self):
        ''' Return a dataframe with indexes of reactions which have been chosen'''
        
        return self.df.loc[self.explored_space_index]
    
    def update_explored_rxns(self, rxn_idxs):
        ''' Add rxn_idxs to the list of explored reactions'''
        self.explored_space_index += rxn_idxs
        
    def is_empty(self):
        ''' Return True if the whole chemical space has been explored'''
        if len(self.explored_space_index) == self.space_len:
            return True
        else:
            return False
        
    def percent_explored(self):
        ''' Return the percent of exploration of chemical space'''
        return len(self.explored_space_index)/self.space_len


class Simulation():
    def __init__(self, data):
        self.data = data
        self.chemspace = ChemicalSpace(data, seed=555)
        self.random_percent = 0.1
    
    def explore_space(self):
        # defines the batch size for exploration of chemical space
        screen_size = 100
        neural_regressor = NeuralRegressor()
        # randomly guess select 10 % of chemical space
        random_guess = self.chemspace.random_guess(self.random_percent)
        
        # keep statistics for plotting 
        rxn = [0.5 * len(random_guess)]
        width = [0.95*len(random_guess)]
        
        # Evaluate average yield of selected reactions
        avg_real_yield = [np.mean(random_guess['Product_Yield_PCT_Area_UV'])]
        # Evaluate standard deviation 
        stds = [np.std(random_guess['Product_Yield_PCT_Area_UV'])]
        prediction_loss = []
        
        
        iteration = 0
        while not self.chemspace.is_empty():
            # Add data to plot
            if iteration == 0:
                rxn.append(2*rxn[-1]+0.5 *screen_size)
            else:
                if len(self.chemspace.get_unused()) < screen_size:
                    screen_size = len(self.chemspace.get_unused())
                rxn.append(rxn[-1]+screen_size)
                
            width.append(0.95*screen_size)
            
            
            # split data into training  and validation set  for reactions explored so far
            train, validation, _ = train_validation_test_split(self.chemspace.get_explored(),
                                                              train_size=0.8, 
                                                              validation_size=0.2,
                                                              seed=123)
            # train the neural network on this data
            neural_regressor.train_model(DataManager(train, data_fields, unique_fields_list),
                                         DataManager(validation, data_fields, unique_fields_list))
            
            # Get a dataframe with all reaction which han't been performed
            unseen = self.chemspace.get_unused()
            # Make make predictions for them and evaluate loss
            yp, loss = neural_regressor.predict(DataManager(unseen, data_fields, unique_fields_list))
            # collect data for statistics
            prediction_loss.append(loss)
            # Add predcited yields to the dataframe 
            unseen['Predicted yield'] = yp
            # Sort the dataframe by predicted yield
            sorted_by_prediction = unseen.sort_values(by=['Predicted yield'],
                                                      ascending=False)
            # Get a dataframe with best candidates from prediction
            best_results = sorted_by_prediction.head(screen_size)
            # Get idxs of best candidates 
            best_results_idxs = list(best_results.index)
            # Evaluate real yiled of selected batch of reactions
            avg_real_yield.append(np.mean(best_results['Product_Yield_PCT_Area_UV']))
            print ('Selected average yield', np.mean(best_results['Product_Yield_PCT_Area_UV']))
            # Calcuate and print the standard deviation for current batch of reactions
            stds.append(np.std(best_results['Product_Yield_PCT_Area_UV']))
            print ('Selected yield std', np.std(best_results['Product_Yield_PCT_Area_UV']))
            # Add the current batch of reactions to explored reactions so in 
            # the next iteration neural network can be trained on updated data
            self.chemspace.update_explored_rxns(best_results_idxs)
            iteration += 1
            # Print some statistics
            print ('Iteration {}'.format(iteration))
            print ('Percent explored {:.2f}'.format(self.chemspace.percent_explored()*100))
            
        #plt.plot(rxn, avg_real_yield)
        #plt.show()
        
        # Create a plot with MSE values during exploration of chemical space
        plt.xlabel('Reaction')
        plt.ylabel('Mean squared error')
        plt.title('Mean squared error for prediction of new reactivity')
        plt.plot(rxn[1:], prediction_loss)
        plt.show()
        
        # Create a data frame 
        stat_dataframe = pd.DataFrame()
        stat_dataframe['rxn'] = rxn
        stat_dataframe['avg_real_yield'] = avg_real_yield
        stat_dataframe['std'] = stds
        stat_dataframe.to_csv('Simulation_stats.csv')
        
        # Create histogram
        colors = ['g' for i in rxn]
        colors[0] = 'orange'
        plt.bar(rxn, avg_real_yield, yerr=stds, width=width, color=colors, )
        plt.title('Average yield per batch for Pd coupling')
        plt.xlabel('Reaction Number')
        plt.ylabel('Average Yield [%]')
        
        plt.show()
        
        #return unseen, best_results_idxs
        
        
        
    def test_regressor(self):
        '''Perform split of the data into training, validation, and test set
            and then trains neural network using training and validation sets
            Evaluates performance of the neural network using test set'''
            
        train_set, validation_set, test_set = train_validation_test_split(self.data,
                                              train_size=0.6, validation_size=0.1,
                                              seed=555)
        train = DataManager(train_set, data_fields, unique_fields_list)
        validation = DataManager(validation_set, data_fields, unique_fields_list)
        test = DataManager(test_set, data_fields, unique_fields_list)
        
        neural_regressor = NeuralRegressor()
        neural_regressor.train_model(train, validation)
        yp, loss = neural_regressor.test_model(test)
        mse = np.mean(np.square(yp.reshape([-1])-test.Y))
        mse = np.mean(np.square(yp.reshape([-1])-test.Y))
        print('MSE {}'.format(mse))
        print('RMSE {}'.format(np.sqrt(mse)))
















simulation = Simulation(dataframe)


#training_data, validation_data, test_data = train_validation_test_split(dataframe)
#
#training = DataManager(training_data, data_fields, unique_fields_list)
#validation = DataManager(validation_data, data_fields, unique_fields_list)
#test = DataManager(test_data, data_fields, unique_fields_list)
#
#
#regressor = NeuralRegressor()
#regressor.train_model(training, validation)
#regressor.test_model(test)




