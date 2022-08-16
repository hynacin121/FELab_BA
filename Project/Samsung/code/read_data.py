import pandas as pd


class dataread:    
    def __init__(self, dir):
        self.dir = dir
        self.kr_cap = pd.read_excel(dir + '/MKT_CAP_DATA_KOR_US.xlsx', sheet_name= 'KOR', index_col= 0, parse_dates= True)         
        self.fac_kr = pd.read_excel(dir + '/3factor_kr.xlsx', index_col= 0, parse_dates= True)
        self.kr_price = pd.read_excel(dir + '/PRICE_DATA_KOR_US.xlsx', sheet_name = 'KOR' ,index_col= 0, parse_dates= True)
        
        self.krx_fin = pd.read_csv(dir + '/krx/KRX300금융.csv', index_col = 0 , parse_dates= True,  encoding = 'ISO-8859-1')        #finance
        self.krx_ig = pd.read_csv(dir + '/krx/KRX300산업재.csv', index_col = 0 , parse_dates= True,  encoding = 'ISO-8859-1')       #industrial goods
        self.krx_mat = pd.read_csv(dir + '/krx/KRX300소재.csv', index_col = 0 , parse_dates= True,  encoding = 'ISO-8859-1')        # material
        self.krx_cd = pd.read_csv(dir + '/krx/KRX300자유소비재.csv', index_col = 0 , parse_dates= True,  encoding = 'ISO-8859-1')   #consumer discretionary
        self.krx_it = pd.read_csv(dir + '/krx/KRX300정보기술.csv', index_col = 0 , parse_dates= True, encoding = 'ISO-8859-1')     # it information technology
        self.krx_cs = pd.read_csv(dir + '/krx/KRX300커뮤니케이션서비스.csv', index_col = 0 , parse_dates= True,  encoding = 'ISO-8859-1')  #cs = communication service
        self.krx_cst = pd.read_csv(dir + '/krx/KRX300필수소비재.csv', index_col = 0 , parse_dates= True,  encoding = 'ISO-8859-1')  #cst = consumer staples
        self.krx_hc = pd.read_csv(dir + '/krx/KRX300헬스케어.csv', index_col = 0 , parse_dates= True, encoding = 'ISO-8859-1')     #health care  In [4]:

        self.us_cap = pd.read_excel(dir + '/MKT_CAP_DATA_KOR_US.xlsx', sheet_name= 'US', index_col= 0, parse_dates= True)#
        self.fac_us = pd.read_csv(dir + '/F-F_Research_Data_Factors_daily.csv', index_col= 0, parse_dates= True)/100
        self.industry_us = pd.read_csv(dir + '/industry_index.csv', index_col= 0, parse_dates= True)/100
        self.us_price = pd.read_excel(dir + '/PRICE_DATA_KOR_US.xlsx', sheet_name = 'US' ,index_col= 0, parse_dates= True)
        self.kor =pd.DataFrame()
        self.us = pd.DataFrame()

    def preprocess_krx(self):
        x_kr = self.kor
        krx = pd.concat([self.krx_fin.close, self.krx_ig.close, self.krx_mat.close, self.krx_cd.close, self.krx_it.close, self.krx_cs.close, self.krx_cst.close, self.krx_hc.close], axis = 1)
        krx.columns  = ['fin', 'ig', 'mat', 'cd', 'it', 'cs', 'cst', 'hc']
        krx = krx.sort_index()
        krx = krx[:"2020"]
        self.krx2 = krx
        krx = krx.pct_change().dropna()
        
        krx = pd.concat([krx, x_kr], axis = 1) 
        krx = krx.dropna()
        krx = krx.loc[:,:"hc"]
        
        
        return(krx)

    def preprocess_us(self):
        self.us_cap.columns = ['aapl_cap', 'sp1500_cap', 'sp500_cap', 'sp400_cap', 'sp600_cap']
        no_aapl = self.us_cap.aapl_cap - self.us_cap.sp1500_cap
        no_aapl = no_aapl.pct_change().dropna()
        us_cap1 = self.us_cap.pct_change().dropna()
        self.us_price.columns = ['aapl', 'sp1500', 'sp500', 'sp400', 'sp600']
        us_price1 = self.us_price[['sp500','sp400', 'sp600']].pct_change().dropna()
        factor_us = self.fac_us[['HML', 'SMB']]
        factor_us = factor_us["2010" : "2020"]


        us = pd.concat([us_cap1, us_price1,  no_aapl, factor_us], axis =1)
        us.columns = ['aapl_cap', 'sp1500_cap', 'sp500_cap', 'sp400_cap', 'sp600_cap', 'sp500', 'sp600', 'sp400', 'newmkt', 'HML', 'SMB']
        us = us.dropna()
        self.us = us
        return(us)
    
    def preprocess_ind_us(self):
        industry_us = self.industry_us["2010":"2020"]
        ind2 = industry_us
        industry_us = pd.concat([industry_us,self.us], axis = 1)
        industry_us = industry_us.dropna()
        industry_us = industry_us.loc[ :, :"Other"]
        
        ind2 = industry_us
        ind2 = ind2 + 1
        ind3 = pd.DataFrame(columns = ind2.columns, index = ind2.index)
        ind3.iloc[0,:] = 1
        
        for j in range(len(ind3.columns)):
            for i in range(len(ind3)-1):
                ind3.iloc[i+1,j] = ind3.iloc[i,j]* ind2.iloc[i+1,j]
                       
        self.indus= ind3.astype(float)     
                
        return(industry_us)

    def preprocess_kr(self):
                
        self.kr_cap.columns = ['k_cap', 'k_l_cap', 'k_m_cap', 'k_s_cap', 'sam_cap']
        no_sam = self.kr_cap.k_cap - self.kr_cap.sam_cap
        no_sam = no_sam.pct_change().dropna()
        kr_cap1 = self.kr_cap.pct_change().dropna()

        
        factor_kr = self.fac_kr[['HML', 'SMB']]
        factor_kr1 = factor_kr.pct_change().dropna()


        self.kr_price.columns = ['sam', 'kospi', 'kospi_large', 'kospi_mid', 'kospi_small']
        kr_price1 = self.kr_price.pct_change().dropna()
 
        kor = pd.concat([kr_price1, kr_cap1, no_sam , factor_kr1], axis = 1)
        kor.columns = ['samsung','kospi', 'kospi_l', 'kospi_m', 'kospi_s', 'k_cap', 'kl_cap', 'km_cap', 'ks_cap', 'sam_cap', 'newmkt', 'HML', 'SMB']
        kor = kor.dropna()
        kor = kor["2010":]


        self.kor = kor
        return(kor)

    def mom1m_kr(self):
        self.kr_price.columns = ['sam', 'kospi', 'kospi_large', 'kospi_mid', 'kospi_small']
        x_kr = self.kor
        mom = pd.concat([self.kr_price[['kospi_large', 'kospi_mid', 'kospi_small']],self.krx2 ], axis = 1)
        mom.columns  = ['kl_mom1', 'km_mom1', 'ks_mom1','fin_mom1', 'ig_mom1', 
                        'mat_mom1', 'cd_mom1', 'it_mom1', 'cs_mom1', 'cst_mom1', 'hc_mom1' ]
        mom = pd.concat([mom, x_kr], axis = 1)
        mom = mom.dropna()
        mom = mom.loc[:,:"hc_mom1"]
        mom = mom["2010-12-03":]
        mom = mom.pct_change(periods = 20).dropna()
        return(mom)
    
    def mom6m_kr(self):
        self.kr_price.columns = ['sam', 'kospi', 'kospi_large', 'kospi_mid', 'kospi_small']
        x_kr = self.kor        
        mom = pd.concat([self.kr_price[['kospi_large', 'kospi_mid', 'kospi_small']],self.krx2 ], axis = 1)
        mom.columns  = ['kl_mom6', 'km_mom6', 'ks_mom6','fin_mom6', 'ig_mom6', 
                        'mat_mom6', 'cd_mom6', 'it_mom6', 'cs_mom6', 'cst_mom6', 'hc_mom6' ]
        mom = pd.concat([mom, x_kr], axis = 1)
        mom = mom.dropna()
        mom = mom.loc[:,:"hc_mom6"] 
        mom = mom["2010-07-04":]
        mom1 = mom.pct_change(periods = 104).dropna()[:"2020-12-01"]
        mom1.index = mom[" 2011":"2020"].index
       

        return(mom1)

    def mom12m_kr(self):
        self.kr_price.columns = ['sam', 'kospi', 'kospi_large', 'kospi_mid', 'kospi_small']
        x_kr = self.kor           
        mom = pd.concat([self.kr_price[['kospi_large', 'kospi_mid', 'kospi_small']],self.krx2 ], axis = 1)
        mom.columns  = ['kl_mom12', 'km_mom12', 'ks_mom12', 'fin_mom12', 'ig_mom12', 
                        'mat_mom12', 'cd_mom12', 'it_mom12', 'cs_mom12', 'cst_mom12', 'hc_mom12' ]
        mom = pd.concat([mom, x_kr], axis = 1)
        mom = mom.dropna()
        mom = mom.loc[:,:"hc_mom12"]  
        mom = mom["2010-01-04":]
        mom1 = mom.pct_change(periods = 222).dropna()[:"2020-12-01"]
        mom1.index = mom["2011":"2020"].index

        return(mom1)

    def chmom_kr(self):
        self.kr_price.columns = ['sam', 'kospi', 'kospi_large', 'kospi_mid', 'kospi_small'] 
        x_kr = self.kor       
        chmom = pd.concat([self.kr_price[['kospi_large', 'kospi_mid', 'kospi_small']],self.krx2 ], axis = 1)
        chmom.columns  = ['kl_chmom', 'km_chmom', 'ks_chmom', 
                          'fin_chmom', 'ig_chmom', 'mat_chmom', 'cd_chmom', 'it_chmom', 'cs_chmom', 'cst_chmom', 'hc_chmom']
        chmom = pd.concat([chmom, x_kr], axis = 1)
        chmom = chmom.dropna()
        chmom = chmom.loc[:,:"hc_chmom"] 
        chmom = chmom["2010-01-01":]
        df = pd.DataFrame()
        for i in range(chmom["2011":"2020"].shape[0]):    
            df = pd.concat([df, pd.DataFrame(chmom.iloc[117+i,]/chmom.iloc[0+i,] -  chmom.iloc[242+i,]/chmom.iloc[118+i,]).transpose()],ignore_index= True)
        df.index = chmom["2011":"2020"].index

        # 7/3 ,  1/4 , 7/4, 12/31
        return(df)

    def maxret_kr(self):
        self.kr_price.columns = ['sam', 'kospi', 'kospi_large', 'kospi_mid', 'kospi_small']  
        x_kr = self.kor
        maxret = pd.concat([self.kr_price[['kospi_large', 'kospi_mid', 'kospi_small']],self.krx2 ], axis = 1)
        maxret.columns  = ['kl_mrt', 'km_mrt', 'ks_mrt',
                           'fin_mrt', 'ig_mrt', 'mat_mrt', 'cd_mrt', 'it_mrt', 'cs_mrt', 'cst_mrt', 'hc_mrt']
        maxret = maxret.pct_change().dropna()
        maxret = pd.concat([maxret, x_kr], axis = 1)
        maxret = maxret.dropna()
        maxret = maxret.loc[:,:"hc_mrt"]
        maxret = maxret["2010-12-03":]
        maxretnew = pd.DataFrame(columns = maxret.columns, index = maxret["2011":].index)
        for j in range(len(maxretnew.columns)):
            for i in range(len(maxretnew)):
                maxretnew.iloc[i,j] = maxret.iloc[i:i+20,j].max()
        maxretnew = maxretnew.astype(float)            
        return(maxretnew)

    def indmom_kr(self):
        x_kr = self.kor
        krx = pd.concat([self.krx_fin.close, self.krx_ig.close, self.krx_mat.close, self.krx_cd.close, self.krx_it.close, self.krx_cs.close, self.krx_cst.close, self.krx_hc.close], axis = 1)
        krx.columns  = ['fin', 'ig', 'mat', 'cd', 'it', 'cs', 'cst', 'hc']
        krx = krx.sort_index()
        krx = krx[:"2020"]               
        indmom = krx.mean(axis = 1)
        indmom.name = "indmom"
        indmom = indmom.pct_change(periods = 251).dropna()
        indmom = pd.concat([indmom, x_kr], axis = 1)
        indmom = indmom.dropna()
        indmom = indmom.loc[:,'indmom']
        indmom.index = x_kr["2011":"2020"].index
        
        return(indmom)
            
    def mom1m_us(self):
        self.us_price.columns = ['aapl', 'sp1500', 'sp500', 'sp400', 'sp600']
        mom = pd.concat([self.us_price[['sp500', 'sp400', 'sp600']], self.indus], axis = 1)
        mom.columns = ['sp5_mom1','sp4_mom1', 'sp6_mom1',
                       'NoDur_mom1', 'Durbl_mom1', 'Manuf_mom1', 'Enrgy_mom1', 'HiTec_mom1', 'Telcm_mom1', 'Shops_mom1', 'Hlth_mom1',
                       'Utils_mom1', 'Other_mom1']
        x_us = self.us
        mom = pd.concat([mom, x_us], axis = 1)
        mom = mom.dropna()
        mom = mom.loc[:,:"Other_mom1"]        
        mom1 = mom["2010-12-04":]
        mom1 = mom1.pct_change(periods = 19).dropna()

        return(mom1)

    def mom6m_us(self):
        self.us_price.columns = ['aapl', 'sp1500', 'sp500', 'sp400', 'sp600']
        mom = pd.concat([self.us_price[['sp500', 'sp400', 'sp600']], self.indus], axis = 1)
        mom.columns = ['sp5_mom6','sp4_mom6', 'sp6_mom6',
                       'NoDur_mom6', 'Durbl_mom6', 'Manuf_mom6', 'Enrgy_mom6', 'HiTec_mom6', 'Telcm_mom6', 'Shops_mom6', 'Hlth_mom6',
                       'Utils_mom6', 'Other_mom6']
        x_us = self.us
        mom = pd.concat([mom, x_us], axis = 1)
        mom = mom.dropna()
        mom = mom.loc[:,:"Other_mom6"]                
        mom1 = mom["2010-07-02":"2020-12-04"]
        mom1 = mom1.pct_change(periods = 105).dropna()
        mom1.index = mom["2011":"2020"].index
        return(mom1)

    def mom12m_us(self):
        self.us_price.columns = ['aapl', 'sp1500', 'sp500', 'sp400', 'sp600']
        mom = pd.concat([self.us_price[['sp500', 'sp400', 'sp600']], self.indus], axis = 1)
        mom.columns = ['sp5_mom12','sp4_mom12', 'sp6_mom12',
                       'NoDur_mom12', 'Durbl_mom12', 'Manuf_mom12', 'Enrgy_mom12', 'HiTec_mom12', 'Telcm_mom12', 'Shops_mom12', 'Hlth_mom12',
                       'Utils_mom12', 'Other_mom12']
        x_us = self.us
        mom = pd.concat([mom, x_us], axis = 1)
        mom = mom.dropna()
        mom = mom.loc[:,:"Other_mom12"]                
        mom1 = mom["2010-01-03":]
        mom1 = mom1.pct_change(periods = 221).dropna()[:"2020-12-02"]
        mom1.index = mom["2011":"2020"].index
        return(mom1)
    
    def chmom_us(self):
        self.us_price.columns = ['aapl', 'sp1500', 'sp500', 'sp400', 'sp600']
        chmom = pd.concat([self.us_price[['sp500', 'sp400', 'sp600']], self.indus], axis = 1)
        chmom.columns = ['sp5_chmom','sp4_chmom', 'sp6_chmom',
                         'NoDur_chmom', 'Durbl_chmom', 'Manuf_chmom', 'Enrgy_chmom', 'HiTec_chmom', 'Telcm_chmom', 'Shops_chmom', 'Hlth_chmom',
                       'Utils_chmom', 'Other_chmom']
        x_us = self.us
        chmom = pd.concat([chmom, x_us], axis = 1)
        chmom = chmom.dropna()
        chmom = chmom.loc[:,:"Other_chmom"]        
        chmom = chmom["2010-01-01":]
        df = pd.DataFrame()
        for i in range(chmom["2011":"2020"].shape[0]):    
            df = pd.concat([df, pd.DataFrame(chmom.iloc[119+i,]/chmom.iloc[0+i,] -  chmom.iloc[240+i,]/chmom.iloc[120+i,]).transpose()],ignore_index= True)
        df.index = chmom["2011":"2020"].index

        return(df)
        
    def maxret_us(self):
        self.us_price.columns = ['aapl', 'sp1500', 'sp500', 'sp400', 'sp600']
        maxret = pd.concat([self.us_price[['sp500', 'sp400', 'sp600']], self.indus], axis = 1)
        maxret.columns = ['sp5_mrt','sp4_mrt', 'sp6_mrt',
                         'NoDur_mrt', 'Durbl_mrt', 'Manuf_mrt', 'Enrgy_mrt', 'HiTec_mrt', 'Telcm_mrt', 'Shops_mrt', 'Hlth_mrt',
                       'Utils_mrt', 'Other_mrt']
        x_us = self.us
        maxret = pd.concat([maxret, x_us], axis = 1)
        maxret = maxret.dropna()
        maxret = maxret.loc[:,:"Other_mrt"]            
        maxret = maxret["2010-12-03":]
        maxretnew = pd.DataFrame(columns = maxret.columns, index = maxret["2011":].index)
        for j in range(len(maxretnew.columns)):
            for i in range(len(maxretnew)):
                maxretnew.iloc[i,j] = maxret.iloc[i:i+18,j].max()
        maxretnew = maxretnew.astype(float)            
        return(maxretnew)
    
    def indmom_us(self):    
        industry_us = self.industry_us["2010":"2020"]
        industry_us = pd.concat([industry_us,self.us], axis = 1)
        industry_us = industry_us.dropna()
        industry_us = industry_us.loc[ :, :"Other"]
        indmom = industry_us.mean(axis = 1)
        indmom2 = indmom/100 + 1
        indmom3 = pd.Series(index = indmom.index)
        indmom3[0] = indmom2[0]
        for i in range(len(indmom)-1):
            indmom3[i+1] = indmom3[i]* indmom2[i+1]
        indmom3 = indmom3.pct_change(periods = 241).dropna()
        indmom3.name = "indmom"
        i
        return(indmom3)
        
