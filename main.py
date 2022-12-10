import numpy as np
import pandas as pd


from QuantConnect.Data.Custom import Quandl
from QuantConnect.Python import PythonQuandl
from QuantConnect.Data.Custom.USTreasury import *
from AlgorithmImports import *

class QuandlAlgo(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2017, 1, 1, 9, 0)  # Set Start Date
        self.SetEndDate(2018, 1, 1, 9, 0)  # Set Start Date
        self.SetCash(1000000)  # Set Strategy Cash
        
        self.pairs =[['MS', 'XOM'], ['GOOG', 'AAPL']]# just a random stock selection
        self.symbols =[]
        
        for ticker in self.pairs:
            
            self.AddEquity(ticker, Resolution.minute)
            self.symbols.append(self.Symbol(ticker))
        
        # drawdown limit = 20%
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.2))

        self.riskfree_rate = self.AddData(TradingEconomicsCalendar, TradingEconomics.Calendar.UnitedStates.GovernmentBondTenY)
        self.lookback = 20 # looback past 20 days

        self.invested = [] # used to remember all the invested pairs. also used for trading cost control
        self.reenter = []
        self.dic = {}
        for each_pair in self.pairs:
            self.invested.append(each_pair[0])
            self.invested.append(each_pair[1])

            #self.dic is used for control trading costs
            if not each_pair[0] in self.dic:
                self.dic[each_pair[0]] = 1
            else:
                self.dic[each_pair[0]] += 1

            if not each_pair[1] in self.dic:
                self.dic[each_pair[1]] = 1
            else:
                self.dic[each_pair[1]] += 1    
        
        self.vix = self.AddIndex('VIX', Resolution.Minute)

    def stats(self, symbols):
        
        #Use Statsmodels package to compute linear regression and ADF statistics

        self.df = self.History(symbols, self.lookback)
        self.dg = self.df["open"].unstack(level=0)
        self.dh = self.df['close'].unstack(level = 0)
        
        #self.Debug(self.dg)
        
        ticker1= str(symbols[0])
        ticker2= str(symbols[1])
        
        
        initial_value = self.dg[ticker1][0] * symbols[2] + self.dg[ticker2][0] * symbols[3]
        current_value = symbols[4] * symbols[2] + symbols[5] * symbols[3]
        mean_return = current_value / initial_value - 1
        
        ticker1_std = np(self.dg([ticker1], 20, Resolution.Daily).pct_change()).std()
        ticker2_std = np(self.dg([ticker2], 20, Resolution.Daily).pct_change()).std()

        cov_matrix = np.cov(self.dh.dropna())
        std = symbols[4] * symbols[4]*ticker1_std * ticker1_std + symbols[5] * symbols[5]*ticker2_std * ticker2_std + 2* symbols[4] * symbols[5]*ticker1_std * ticker2_std * cov_matrix

        # to compute z-score
        Y = self.dg[ticker1].apply(lambda x: math.log(x))
        X = self.dg[ticker2].apply(lambda x: math.log(x))
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
        sigma = math.sqrt(results.mse_resid) # standard deviation of the residual
        slope = results.params[1]
        intercept = results.params[0]
        res = results.resid #regression residual mean of res =0 by definition
        zscore = res/sigma
        adf = adfuller (res)
        
        return [mean_return, std, zscore]
    






    def OnData(self, data: Slice):
        # our strategy is here
        # buying amounts determined by our trading strategy. here is a random number. not this weight is the single stock to total portfolio

        buying_weight_0 = 0.15
        buying_weight_1 = 0.1
        
        #for re-enter
        if self.reenter:
            for dropped_pair in self.reenter:
                
                dropped_stats = [dropped_pair[0], dropped_pair[1], buying_weight_0, buying_weight_1, self.Securities[pair[0]].Price, self.Securities[pair[1]].Price]
                zscore_pair = self.stats(dropped_stats)[2]
                if zscore_pair > -2 and zscore_pair < 2 and self.vix < 33:
                    self.reenter.remove(dropped_pair)
                    self.pairs.append(dropped_pair)

        # make purchase
        for pair in self.pairs:
            
            # buying amount limit

            var_weight = ### VAR limit 
            symbols = [pair[0], pair[1], buying_weight_0, buying_weight_1, self.Securities[pair[0]].Price, self.Securities[pair[1]].Price]
            mean_return, std = self.stats(symbols)[0], self.stats(symbols)[1]
        
            kelly_weight = (mean_return - self.riskfree_rate)/ std
            ### use the link below for live trading code
            # https://www.quantconnect.com/forum/discussion/7730/what-is-a-good-soure-for-risk-free-rate-working-in-live/p1

            #margin = self.Portfolio.GetMarginRemaining(pair[0], OrderDirection.Buy)
            margin = self.Portfolio.MarginRemaining
            margin_used = self.Portfolio.TotalMarginUsed
            
            margincall_weight = margin / (margin + margin_used)
            h_threshold = min(var_weight, margincall_weight)
            # l_threshold = kelly_weight or how to use kelly's principle
            if h_threshold >= buying_weight and not self.Portfolio.Invested and self.dic[pair[0]] > 0 and self.dic[pair[1]] > 0: # self.dic is used for trading costs
                self.SetHoldings(pair[0], buying_weight_0)
                self.SetHoldings(pair[1], buying_weight_1)
                self.dic[pair[0]] -= 1
                self.dic[pair[1]] -= 1
               

        
        # risk management
        
        #[x.Key for x in self.Portfolio if x.Value.Invested] 
        # self.pair is the list that contains all the pairs we hold currently

        # daily loss limit
        for pair1 in self.pairs:
            stock1 = pair1[0]
            stock2 = pair1[1]
            stock1_open = self.Securities[stock1].open
            stock2_open = self.Securities[stock2].open
            stock1_quantity = self.Portfolio[stock1].Quantity
            stock2_quantity = self.Portfolio[stock2].Quantity
        
            current_pair_value = self.Securities[stock1].Price * stock1_quantity + self.Securities[stock2].Price * stock2_quantity
            base_pair_value = stock1_open * stock1_quantity + stock2_open * stock2_quantity
            if current_pair_value/base_pair_value < 0.95: ### modify here
                self.Liquidate(self.Securities[stock1])
                self.Liquidate(self.Securities[stock2])

                if [stock1, stock2] in self.pairs:
                    self.pairs.remove([stock1, stock2])
                    self.reenter.append([stock1, stock2])
                elif [stock2, stock1] in self.pairs:
                    self.pairs.remove([stock2, stock1])
                    self.reenter.append([stock2, stock1])
