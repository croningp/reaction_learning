import random
import numpy as np

def train_validation_test_split(df, train_size=0.6,
                                validation_size=.1, seed=None):
    """ Split data into train, validation, and test set """
    
    random.seed(seed)
    index = list(df.index)
    random.shuffle(index)
    #print(index[0:10])
    df = df.loc[index]
    df = df.reset_index()
    data_len = len(index)
    train_idx = int(data_len*train_size)
    validation_idx = int(data_len*validation_size) + train_idx
    #print (train_idx,validation_idx)
    train_data_set = df.loc[range(train_idx)]
    validation_set = df.loc[range(train_idx,validation_idx)]
    test_set = df.loc[range(validation_idx, data_len)]
    
    return train_data_set, validation_set, test_set


class DataManager():
    ''' A class for one-hot encoding of the data and generating batches 
        for tensorflow'''
    
    def __init__(self, df, data_fields, unique_fields_list):
        self.df = df
        self.X = self.encode(self.df, data_fields, unique_fields_list)
        self.Y = self.df['Product_Yield_PCT_Area_UV'].as_matrix()
        # create numpy array and scale to range 0.0 to 1.0 
        self.Y = np.array(self.Y) / 100
        self.curr_idx= 0
        
        
        
    def next_batch(self, batch_size):
        x = self.X[self.curr_idx: self.curr_idx+batch_size]
        y = self.Y[self.curr_idx: self.curr_idx+batch_size]
        
        self.curr_idx = (self.curr_idx + batch_size) % len(self.X)
        if x.shape[1] !=37:
            raise ValueError("Wrong shape")
        return x, y
    
    def encode(self, df, data_fields, unique_fields_list):
        def one_hot_encode(data, labels):
            #print (labels)
            #print (data[0:10])
            n = len(data)
            n_labels = len(labels)
            idxs = [labels.index(l) for l in data]
            one_hot_encoded =   np.zeros([n, n_labels])
            one_hot_encoded[range(n), idxs] = 1
            return one_hot_encoded
            
        X = [df[field].as_matrix() for field in data_fields]
        X_one_hot = list(map(one_hot_encode, X, unique_fields_list))
        return np.concatenate(X_one_hot, axis=1)