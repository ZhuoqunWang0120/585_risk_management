import numpy as np
import pandas as pd
from scipy.stats import norm


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
            
            self.AddEquity(ticker, Resolution.Minute)
            self.symbols.append(self.Symbol(ticker))
        
        # drawdown limit = 20%
        self.SetRiskManagement(MaximumDrawdownPercentPerSecurity(0.2))

        self.riskfree_rate = self.AddData(TradingEconomicsCalendar, TradingEconomics.Calendar.UnitedStates.GovernmentBondTenY)
        self.lookback = 20 # looback past 20 days

        # VaR 
        self.lookback_VaR = 1*24*60
        self.portfolio_val = []
        self.count = 0
        self.VaR_limit = -0.05 # probably too restrictive. may want to change to something like 0.1 to reduce its effect on the algo

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
    def VaR_normal(mu, sigma, c = 0.95):
        """
        Variance-Covariance calculation (with Gaussian assumption) of Value-at-Risk
        using confidence level c (e.g., 0.95), with mean of returns mu
        and standard deviation of returns sigma
        Returns the log-return l where P(return > l) = c
        """
        r = norm.ppf(1-c, mu, sigma)
        return r
    def VaR_historical(rets, c = 0.95):
        """
        Calculate value-at-Risk with confidence level c (e.g., 0.95) based on historical return rets (list)
        Returns the log-return l where P(return > l) = c
        """
        r = sorted(rets)[max(round(len(rets) * (1 - c)) - 1, 0)]
        return r
    def VaR(rets, c = None, method = 'normal'):
        if not c:
            c = 0.95
        if method == 'historical':
            return VaR_historical(rets, c)
        else:
            mu = np.mean(rets)
            sigma = np.sqrt(np.var(rets))
            return VaR_normal(mu, sigma, c)
    
    
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

            # var_weight = abs(self.VaR_limit) ### VAR limit 
            symbols = [pair[0], pair[1], buying_weight_0, buying_weight_1, self.Securities[pair[0]].Price, self.Securities[pair[1]].Price]
            mean_return, std = self.stats(symbols)[0], self.stats(symbols)[1]
        
            kelly_weight = (mean_return - self.riskfree_rate)/ std
            ### use the link below for live trading code
            # https://www.quantconnect.com/forum/discussion/7730/what-is-a-good-soure-for-risk-free-rate-working-in-live/p1

            #margin = self.Portfolio.GetMarginRemaining(pair[0], OrderDirection.Buy)
            margin = self.Portfolio.MarginRemaining
            margin_used = self.Portfolio.TotalMarginUsed
            
            margincall_weight = margin / (margin + margin_used)
            # h_threshold = min(var_weight, margincall_weight)
            h_threshold = margincall_weight
            # l_threshold = kelly_weight or how to use kelly's principle
            if h_threshold >= buying_weight and self.dic[pair[0]] > 0 and self.dic[pair[1]] > 0: # self.dic is used for trading costs
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

        # VaR
        self.portfolio_val.append(self.Portfolio.TotalPortfolioValue)
        if len(self.portfolio_val) > self.lookback_VaR:
            self.portfolio_val.pop(0)
        if len(self.portfolio_val) >=30:
            logval = np.log(np.array(self.portfolio_val))
            rets = logval[1:] - logval[-1]
            VaR_stats = VaR(rets)
            if VaR_stats < self.VaR_limit: 
                k = VaR_limit/VaR_stats
                k = min(k,1)
                k = max(k,0)
                for pair1 in self.pairs:
                    stock1 = pair1[0]
                    stock2 = pair1[1]
                    stock1_pct = abs(self.Portfolio[stock1].HoldingsValue/(10**7))
                    self.SetHoldings(stock1, stock1_pct * k)
                    stock2_pct = abs(self.Portfolio[stock2].HoldingsValue/(10**7))
                    self.SetHoldings(stock2, stock1_pct * k)

        self.count += 1

        if self.count % (6.5*60) == 0:
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
