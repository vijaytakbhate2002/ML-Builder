import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score

class Builder():
    def __init__(self,df,algo,problem_type,target_name,cross_scores,solve_overfit=False):
        self.df = df
        self.model = algo
        self.problem_type = problem_type
        self.target_name = target_name
        self.cross_scores = cross_scores
        self.solve_overfit = solve_overfit
        
    def fit_make(self):
        return self.make()
    
    class HeadProcess():
        def __init__(self,df,target_name,problem_type=None):
            self.df = df
            self.target_name = target_name
            self.problem_type = problem_type
            
        def separate_cat_num(self,null_cols):
            """separating categorical and numerical values"""
            num_df = self.df.select_dtypes(include=[np.number])
            cat_df = self.df.select_dtypes(exclude=[np.number])
            """recognizing problem type (classification or Regression)"""
            if self.target_name in cat_df.columns:
                self.problem_type = "class"
                target_df = cat_df[self.target_name]
                cat_df.drop(self.target_name,axis='columns',inplace=True)
            else:
                self.problem_tyep = "reg"
                target_df = num_df[self.target_name]
                num_df.drop(self.target_name,axis='columns',inplace=True)
            return cat_df,num_df,target_df,self.problem_type,null_cols
            
        def null_killer(self):
            null_dic = dict(self.df.isnull().sum())
            null_cols = []
            vals = list(null_dic.values())
            keys = list(null_dic.keys())
            while max(vals) != 0:
                if len(self.df)//2 <= max(vals):
                    self.df.drop([keys[np.argmax(vals)]],axis='columns',inplace=True)
                    null_cols.append(keys[np.argmax(vals)])
                    del(vals[np.argmax(vals)])
                    del(keys[np.argmax(vals)])
                else:
                    self.df.dropna(inplace=True)
                    vals = [0]*len(vals)
            if "Unnamed: 0" in self.df.columns:
                null_cols.append("Unnamed: 0")
                self.df.drop("Unnamed: 0",axis=1,inplace=True)
            return self.separate_cat_num(null_cols)
            
    class NumProcess():
        def __init__(self,df,solve_overfit):
            self.df = df
            self.solve_overfit = solve_overfit
            
        def standard_scaler(self):
            if self.solve_overfit:
                self.df = self.feature_adder()
            scalar = StandardScaler()
            arr = scalar.fit_transform(self.df)
            self.df = pd.DataFrame(arr,columns=self.df.columns)
            return self.df
        
        def minmax_scaler(self):
            if self.solve_overfit:
                self.feature_adder()
            scalar = MinMaxScaler()
            arr = scalar.fit_transform(self.df)
            self.df = pd.DataFrame(arr,columns=self.df.columns)
            return self.df    
        
        def feature_adder(self):
            for col in self.df.columns:
                self.df[col+"_extra"] = self.df[col]*self.df[col] * self.df[col]
    
    class CatProcess():
        def __init__(self,df):
            self.df = df

        def label_encoding(self):
            encoder = LabelEncoder()
            if type(self.df) == pd.core.series.Series:
                self.df = encoder.fit_transform(self.df)
            else:
                for col in self.df.columns:
                    self.df[col] = encoder.fit_transform(self.df[col])
            return self.df
        
    class TailProcess():
        def __init__(self,X,y,model,cross_scores):
            self.X = X
            self.y = y
            self.model = model
            self.cross_scores = cross_scores
        
        def train(self,null_cols):
            X_train, X_test, y_train, y_test = train_test_split(self.X,self.y,test_size=0.3)
            model_obj = self.model
            dum = model_obj.fit(X_train,y_train)
            train_score = model_obj.score(X_train,y_train)
            test_score = model_obj.score(X_test,y_test)
            cross_scores = "Disabled"
            if self.cross_scores:
                cross_scores = cross_val_score(model_obj,self.X,self.y)
            model = model_obj.fit(self.X,self.y)
            X_train, X_test, y_train, y_test = pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)
            model_kit = {"Used algo":model_obj,"test_score":test_score,'train_score':train_score,"cross_scores":cross_scores,"Droped Columns":null_cols,"X_train":X_train,"y_train":y_train,"X_test":X_test,"y_test":y_test}
            return model_kit,model_obj
    
    def x_processor(self,cat_df,num_df,empty_df):
        if empty_df == "cat_df":
            num_process = self.NumProcess(num_df,self.solve_overfit)
            full_df = num_process.minmax_scaler()
        elif empty_df == "num_df":
            cat_process = self.CatProcess(cat_df)
            full_df = cat_process.label_encoding()
        else:
            cat_process = self.CatProcess(cat_df)
            num_process = self.NumProcess(num_df,self.solve_overfit)
            cat_df = cat_process.label_encoding()
            num_df = num_process.minmax_scaler()
            cat_df.reset_index(drop=True,inplace=True)
            num_df.reset_index(drop=True,inplace=True)
            full_df = pd.concat([cat_df,num_df],axis='columns')
        return full_df
    
    def target_process(self,target_df):
        if self.problem_type == "class":
            return self.CatProcess(target_df).label_encoding()
        else:
            return target_df
    
    def make(self):
        head_processor = self.HeadProcess(self.df,self.target_name)
        cat_df,num_df,target_df,problem_type,null_cols = head_processor.null_killer() 
        target_df = self.target_process(target_df)   
        if num_df.empty:
            full_df = self.x_processor(cat_df,num_df,empty_df="num_df")
        elif cat_df.empty:
            full_df = self.x_processor(cat_df,num_df,empty_df="cat_df")
        else:
            full_df = self.x_processor(cat_df,num_df,empty_df=None)
            
        return self.TailProcess(full_df,target_df,self.model,self.cross_scores).train(null_cols)

