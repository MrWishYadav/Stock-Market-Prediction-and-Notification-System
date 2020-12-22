from nsetools import Nse

class NSE:
    def __init__(self):
        self.NSE = Nse()
    
    def getNSECompanyNames(self):
        NSECompanyCodes = self.NSE.get_stock_codes(cached=False)
        NSECompanyList = list(NSECompanyCodes.keys())[1:]
        return NSECompanyList
