import pandas as pd


class dataread:    
    def __init__(self, dir):
        self.dir = dir
        self.pfo0 = pd.read_excel(dir + '/result_pfo_ret_intensity.xlsx', sheet_name= 'pfo_ret_0.0', index_col= 0, parse_dates= True)
        self.pfo1 = pd.read_excel(dir + '/result_pfo_ret_intensity.xlsx', sheet_name= 'pfo_ret_0.1', index_col= 0, parse_dates= True)
        self.pfo2 = pd.read_excel(dir + '/result_pfo_ret_intensity.xlsx', sheet_name= 'pfo_ret_0.2', index_col= 0, parse_dates= True) 
        self.pfo3 = pd.read_excel(dir + '/result_pfo_ret_intensity.xlsx', sheet_name= 'pfo_ret_0.3', index_col= 0, parse_dates= True) 
        self.pfo4 = pd.read_excel(dir + '/result_pfo_ret_intensity.xlsx', sheet_name= 'pfo_ret_0.4', index_col= 0, parse_dates= True) 
        self.pfo5 = pd.read_excel(dir + '/result_pfo_ret_intensity.xlsx', sheet_name= 'pfo_ret_0.5', index_col= 0, parse_dates= True) 
        self.pfo6 = pd.read_excel(dir + '/result_pfo_ret_intensity.xlsx', sheet_name= 'pfo_ret_0.6', index_col= 0, parse_dates= True) 
        self.pfo7 = pd.read_excel(dir + '/result_pfo_ret_intensity.xlsx', sheet_name= 'pfo_ret_0.7', index_col= 0, parse_dates= True) 
        self.pfo8 = pd.read_excel(dir + '/result_pfo_ret_intensity.xlsx', sheet_name= 'pfo_ret_0.8', index_col= 0, parse_dates= True)  
        self.pfo9 = pd.read_excel(dir + '/result_pfo_ret_intensity.xlsx', sheet_name= 'pfo_ret_0.9', index_col= 0, parse_dates= True)  
        self.pfo10= pd.read_excel(dir + '/result_pfo_ret_intensity.xlsx', sheet_name= 'pfo_ret_1.0', index_col= 0, parse_dates= True) 
        
        
        self.ind_pfo0 = pd.read_excel(dir + '/result_ind49.xlsx', sheet_name= 'pfo_ret_0.0', index_col= 0, parse_dates= True)
        self.ind_pfo1 = pd.read_excel(dir + '/result_ind49.xlsx', sheet_name= 'pfo_ret_0.1', index_col= 0, parse_dates= True)
        self.ind_pfo2 = pd.read_excel(dir + '/result_ind49.xlsx', sheet_name= 'pfo_ret_0.2', index_col= 0, parse_dates= True) 
        self.ind_pfo3 = pd.read_excel(dir + '/result_ind49.xlsx', sheet_name= 'pfo_ret_0.3', index_col= 0, parse_dates= True) 
        self.ind_pfo4 = pd.read_excel(dir + '/result_ind49.xlsx', sheet_name= 'pfo_ret_0.4', index_col= 0, parse_dates= True) 
        self.ind_pfo5 = pd.read_excel(dir + '/result_ind49.xlsx', sheet_name= 'pfo_ret_0.5', index_col= 0, parse_dates= True) 
        self.ind_pfo6 = pd.read_excel(dir + '/result_ind49.xlsx', sheet_name= 'pfo_ret_0.6', index_col= 0, parse_dates= True) 
        self.ind_pfo7 = pd.read_excel(dir + '/result_ind49.xlsx', sheet_name= 'pfo_ret_0.7', index_col= 0, parse_dates= True) 
        self.ind_pfo8 = pd.read_excel(dir + '/result_ind49.xlsx', sheet_name= 'pfo_ret_0.8', index_col= 0, parse_dates= True)  
        self.ind_pfo9 = pd.read_excel(dir + '/result_ind49.xlsx', sheet_name= 'pfo_ret_0.9', index_col= 0, parse_dates= True)  
        self.ind_pfo10= pd.read_excel(dir + '/result_ind49.xlsx', sheet_name= 'pfo_ret_1.0', index_col= 0, parse_dates= True)         
        
        
        
        
        
           
        self.fac5 = pd.read_csv(dir + '/F-F_Research_Data_5_Factors_2x3_daily.csv', index_col= 0, parse_dates= True)/100
        
        pfo0 = self.pfo0      

    def preprocess_pfo(self, df):
        df = df["1980":"2021"]
        return(df)


    
    def preprocess_fac5(self):
        fac5 = pd.concat([self.fac5, self.pfo0], axis = 1)["1980":"2021"].dropna()
        fac5 = fac5.loc[:,:"RF"]
        return(fac5)