import pandas as pd
import os
import sys
import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# Adjusts plots
sns.set_style('white')
sns.set_context("poster")
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(10,8))
plt.subplots_adjust( bottom=0.15, right=0.9, top=0.85, left = 0.30, wspace=.4)



def rxn_tovect(rxn, reagents):
    '''Function to convert reaction into its vector representation'''
    r_vect = [0]*len(reagents)
    for r in rxn:
        r_vect[reagents.index(r)] = 1
    return r_vect

def vect_to_rxn(vect, reagents):
    ''' Function to convert vector representation into reaction '''
    
    rxn = []
    for idx, val in enumerate(vect):
        if val==1:
            rxn.append(reagents[idx])

    return rxn

def plot_lda(df):
    ''' Plots lda projection of reaction space '''
    
    
    # set graph formatting 
    sns.set_style('white')
    sns.set_context("poster")
    sns.set_context("notebook", font_scale=1.8, rc={"lines.linewidth": 2.5})
    plt.figure(figsize=(10,8))
    plt.subplots_adjust( bottom=0.15, right=0.9, top=0.9, left = 0.15, wspace=.4)
    # get variables
    X = list(df['X'])
    y = list(df['Reactivity'])
    clf = LDA()
    clf.fit(X, y)
    transformed = clf.transform(X)
    colors = []
    for i in y:
        if i ==0:
            colors.append('b')
        else:
            colors.append('r')
            
    plt.title('LDA projection of reaction space',)# fontsize=30)
    plt.xlabel('LDA score',)# fontsize=25)
    plt.ylabel('LDA score')# fontsize=25)
    #for idx, t in enumerate(transformed):
    #   plt.annotate(str(idx), (t,t))
    plt.scatter(transformed, transformed, s=80, c=colors, )
    path = os.path.join(root_path, 'figures', 'lda.pdf')
    plt.savefig(path, dpi=300, alpha=0.3)
    plt.show()
    plt.close()
    df['LDA_Reactivity'] = transformed
    ldadf = df.sort_values(by=['LDA_Reactivity'], ascending=False)
    
    
    return ldadf


class ChemicalSpace():
    ''' An abstract class alowing to simulate exploration of chemical space
        using algorhitms'''
        
    def __init__(self, dataframe, seed=None):
        
        self.dataframe = dataframe
        self.size = len(dataframe)
        self.explored_space_indexs = []
        self.index = list(range(self.size))
        # makes things random
        random.shuffle(self.index, random=seed)
    
    def random_guess(self, num_of_rxns):
        '''Returns a dataframe with random rxns from chemical space'''
        
        
        random_rxns_idxs = []
        
        for idx in self.index:
            if idx not in self.explored_space_indexs:
                self.explored_space_indexs.append(idx)
                random_rxns_idxs.append(idx)
                
                if len(random_rxns_idxs) ==  num_of_rxns:
                    break
        random_df = self.dataframe.loc[random_rxns_idxs]
        
        return random_df
    
    def get_unused(self):
        ''' Return a dataframe of all reactions which has not been explored so far'''
        
        unused_rxn_idxs = []
        for idx in self.index:
            if idx not in self.explored_space_indexs:
                unused_rxn_idxs.append(idx)
                
        unused_rxns = self.dataframe.loc[unused_rxn_idxs]
        return unused_rxns
    
    def get_explored(self):
        ''' Return a dataframe with indexes of reactions which have been chosen'''
        
        return self.dataframe.loc[self.explored_space_indexs]
    
    def update_explored_rxns(self, rxn_idxs):
        ''' Add rxn_idxs to the list of explored reactions'''
        self.explored_space_indexs += rxn_idxs
        
    def is_empty(self):
        ''' Return True if the whole chemical space has been explored'''
        if len(self.explored_space_indexs) == self.size:
            return True
        else:
            return False
    
    def percent_explored(self):
        ''' Return the percent of exploration of chemical space'''
        return len(self.explored_space_indexs)/self.size
    
    
class Simulation():
    ''' A class simulating exploration of chemical space using machine learning'''
    
    def __init__(self, dataframe, clf):
        self.dataframe = dataframe
        self.clf = clf

    def single_run(self, random_guess=0.1):
        
        chemspace = ChemicalSpace(self.dataframe)
        clf = self.clf()
        num_random_rxns = int(chemspace.size * random_guess)
        
        time_step = []
        num_reactive = []
        num_unreactive = []
        accuracy = []
        
        # random exploration of chemical space
        for r in range(num_random_rxns):
            chemspace.random_guess(1)
            
            data = chemspace.get_explored()
            time_step.append(chemspace.percent_explored()*100)
            
            
            
            num_reactive.append(sum(data['Reactivity']))
            num_unreactive.append(len(data) - sum(data['Reactivity']))
            
            
            unexplored_rxns = chemspace.get_unused()
            reactvity_random_guess = np.random.randint(2, size=len(unexplored_rxns))
            
            accuracy_ = 100*np.sum(np.equal(unexplored_rxns['Reactivity'],
                                      reactvity_random_guess)) / len(unexplored_rxns)
            accuracy.append(accuracy_)
        # Machine learning part
        for i in range(chemspace.size-num_random_rxns):
            
            # Get all reaction selected so fat
            explored_rxns = chemspace.get_explored()
            # Train machine learning model
            clf.fit(list(explored_rxns['X']),
                    list(explored_rxns['Reactivity']))
            
            # Get unexplored reactions
            unexplored_rxns = chemspace.get_unused()
            # Score them using ML
            scores = clf.predict_proba(list(unexplored_rxns['X']))
            # Add scores of being reactive to dataframe 
            # and sort it by probabilty 
            unexplored_rxns['Reaction_proba'] = scores[:,1]
        
            sorted_rxns = unexplored_rxns.sort_values(by=['Reaction_proba'],
                                                      ascending=False)
            
            # Get results with highest proba
            candiate_rxn = sorted_rxns.head(1)
            #print (candiate_rxn['Reaction'], candiate_rxn['Reaction_proba'])
            #print (list(candiate_rxn.index))
            # Get its index
            candidate_idx = list(candiate_rxn.index)
            
            chemspace.update_explored_rxns(candidate_idx)
            
            # update statistics
            if not chemspace.is_empty():
                explored_rxsn = chemspace.get_explored()
                
                time_step.append(chemspace.percent_explored()*100)
                num_reactive.append(sum(explored_rxsn['Reactivity']))
                num_unreactive.append(len(explored_rxsn) - sum(explored_rxsn['Reactivity']))
                
                clf.fit(list(explored_rxsn['X']), list(explored_rxsn['Reactivity']))
                unexplored_rxns = chemspace.get_unused()
                predictions = clf.predict(list(unexplored_rxns['X']))
                
                accuracy_ = np.mean(np.equal(predictions, list(unexplored_rxns['Reactivity']))) *100
                accuracy.append(accuracy_)
                
        return time_step, num_unreactive, num_reactive, accuracy
    
    def run(self, random_guess=0.1, num_iter=3):
        ''' Perform simulation exploring chemical space num_iter'''
        
        # Statistics
        time_step = []
        num_unreactive = []
        num_reactive = []
        accuracy = []
        
        for i in tqdm(range(num_iter)):
            
            i_time_step, i_num_unreactive, i_num_reactive, i_accuracy =\
            self.single_run(random_guess=random_guess)
            
            
            time_step.append(i_time_step)
            num_unreactive.append(i_num_unreactive)
            num_reactive.append(i_num_reactive)
            accuracy.append(i_accuracy)
        
        min_accuracy = np.min(accuracy, axis=0)
        max_accuracy = np.max(accuracy, axis=0)
        mean_accuracy = np.mean(accuracy,axis=0)

        
        
        # random statistics
        r_time_step = []
        r_num_unreactive = []
        r_num_reactive = []
        r_accuracy = []
        
        # Perform simulation wiht random guess
        for i in tqdm(range(num_iter)):
            # randomly keep guessing the whole space
            i_time_step, i_num_unreactive, i_num_reactive, i_accuracy =\
            self.single_run(random_guess=1.0)
            
            
            r_time_step.append(i_time_step)
            r_num_unreactive.append(i_num_unreactive)
            r_num_reactive.append(i_num_reactive)
            r_accuracy.append(i_accuracy)
        
        r_min_accuracy = np.min(r_accuracy, axis=0)
        r_max_accuracy = np.max(r_accuracy, axis=0)
        r_mean_accuracy = np.mean(r_accuracy, axis=0)
        
        
        
        plt.plot(time_step[0], mean_accuracy, color='red')
        plt.fill_between(time_step[0], max_accuracy, min_accuracy, alpha=0.25, color='red')
        plt.plot(r_time_step[0], r_mean_accuracy, color='blue')
        plt.fill_between(r_time_step[0], r_max_accuracy, r_min_accuracy, alpha=0.25, color='blue')
        plt.xlabel('% of space explored')#, fontsize=16)
        plt.xlim(0, 110)
        plt.ylabel('prediction accuracy [%]')# fontsize=16)
        plt.ylim(40, 105)
        plt.title('Reactvity prediction accuracy for\n unexplored reaction space')#, fontsize=17)
        #plt.legend(loc = 2)# fontsize=17)
        
        path = os.path.join(root_path, 'figures', 'accuracy.pdf')
        plt.savefig(path)
        plt.show()
        plt.close()
        
        # Prepeare bar graph comparing reactive and unreactive 
        avg_num_unreactive = np.mean(num_unreactive, axis=0)
        avg_num_reactive = np.mean(num_reactive, axis=0)
        
        
        num_idxs = len(time_step[0])
        idxs = []
        for i in range(1,11):
            fraction = float(i)/10
            idx = int(num_idxs*fraction)-1
            idxs.append(idx)
        width = [9 for i in range(len(idxs))]
        
        bar_space = [time_step[0][i]-5 for i in idxs]
        bar_reactive = [avg_num_reactive[i] for i in idxs]
        bar_unreactive = [avg_num_unreactive[i] for i in idxs]
        scale = [1,2]
        
        plt.xlabel('% of space explored')
        plt.ylabel('total number of mixtures')
        plt.title('Statistics of Reactivity')
        plt.xlim(0,105)
        plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plt.ylim(0, 1000)
        p1 = plt.bar(bar_space, bar_unreactive, color = 'b', width=width)
        p2 = plt.bar(bar_space, bar_reactive, bottom=bar_unreactive, width=width, color='r')
        plt.legend((p1[0], p2[0]), ('Unreactive', 'Reactive'), loc='upper left')
        plt.show()
        path = os.path.join(root_path, 'figures', 'reactvity_stats.pdf')
        plt.savefig(path)
        plt.close()

#        

        
        
    
        


HERE_PATH = os.path.dirname(__file__)
root_path = os.path.join(HERE_PATH, '..')
sys.path.append(root_path)

# open the data as pandas data frame
data_path = os.path.join(root_path, 'data//heterocycles.csv')
dataframe = pd.DataFrame.from_csv(data_path)

# get unique chemicals
allrxns = [rxn.split('_') for  rxn in list(dataframe['Reaction'])]
reagents = list(set([item for sublist in allrxns for item in sublist]))
# Number of chemicals in chemical space
space_len = len(reagents)

X = [rxn_tovect(rxn, reagents) for rxn in allrxns ]
dataframe['X'] = X 

#chemcial_space = ChemicalSpace(dataframe)
s = Simulation(dataframe, LDA)
t = s.run()