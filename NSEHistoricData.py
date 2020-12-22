"""
Created on Mon Sep 21 07:33:12 2020

@author: vishal Gajendrarao Yadav
"""
#This program creates a database of all the listed companies in the National Stock Exchange(NSE) into an excel sheet and updates regularly 
#We are mainly targeting RELIANCE company as an example for our program

from NSE import NSE
from nsepy import get_history
from pathlib import Path
from openpyxl import load_workbook
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import datetime
import time

class NSECompanyHistoricData:
    def __init__(self, directory=str(Path(__file__).resolve().parent)+'/NSEHistoricData/'):
        self.NSECompanyList = NSE().getNSECompanyNames()
        self.directory = directory
        self.companyCount = 0
    
    def getHistory(self, company):
        try:
            reader = None
            NSECompanyHistoricDataFromDate = datetime.datetime.strptime('1990-01-01', '%Y-%m-%d')
            NSECompanyHistoricDataToDate = datetime.datetime.now() + datetime.timedelta(days = 1)
            if(Path(self.directory+company+'.xlsx').exists()):
                reader = pd.read_excel(open(self.directory+company+'.xlsx', 'rb'),sheet_name=company)
                if(len(reader.index) != 0):
                    NSECompanyHistoricDataFromDate = datetime.datetime.strptime(str(reader.iloc[-1,0])[:10], '%Y-%m-%d') + datetime.timedelta(days = 1)
            if((NSECompanyHistoricDataToDate - NSECompanyHistoricDataFromDate) > datetime.timedelta(days = 1)):
                companyHistory = get_history(symbol = company, start = datetime.date(NSECompanyHistoricDataFromDate.year, NSECompanyHistoricDataFromDate.month, NSECompanyHistoricDataFromDate.day), end = datetime.date(NSECompanyHistoricDataToDate.year, NSECompanyHistoricDataToDate.month, NSECompanyHistoricDataToDate.day))
                companyHistory.reset_index(inplace = True)
                writer = pd.ExcelWriter(self.directory+company+'.xlsx', engine='openpyxl')
                if(reader is not None):
                    companyHistory = reader.append(companyHistory, ignore_index=True)
                companyHistory['Date'] = pd.to_datetime(companyHistory['Date'])
                companyHistory.to_excel(writer, sheet_name=company, index=False)
                writer.save()
                writer.close()
            self.companyCount+=1
            print('Company No: %d, Company Name: %s'%(self.companyCount, company))
        except Exception as e:
            print("Exception in getHistory(): " + str(e))

    def saveNSECompanyHistoricData(self):
        try:
            with ThreadPoolExecutor(max_workers = 3) as executor:
                companyHistory = executor.map(self.getHistory, self.NSECompanyList)
                executor.shutdown(wait=True)
        except Exception as e:
            print("Exception in saveNSECompanyHistoricData(): " + str(e))
        
start = time.time()
companyHistoricData = NSECompanyHistoricData(str(Path(__file__).resolve().parent)+'/NSEHistoricData/')
companyHistoricData.saveNSECompanyHistoricData()
print('\nTime Taken(sec): ',(time.time()-start))

