#region imports
from AlgorithmImports import *
import numpy as np
import io
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy import signal
import sys
import dateutil.parser
from collections import deque
import torch
import nltk
import base64
import json
from sklearn import preprocessing
from requests.models import PreparedRequest
from structural_break_model import SBNet
#endregion

class CopulaPairsTradingAlgorithm(QCAlgorithm):
    
    def Initialize(self):
        '''Initialize algorithm and add universe'''

        self.alpaca_key_id = "PKVSYHBMRS0RN8BTWYPB"
        self.alpaca_secret_key = "Vd3fenKG0GSmPOlBfg1Lj8RsO5maVqERUgCqAtZE"

        self.sb_net = SBNet(4, 32, 1)
        model_file = self.Download('https://www.dropbox.com/s/z765zs5xw1gnkhi/base_64_sb_net.pt?dl=1')
        self.sb_net.load_state_dict(torch.load(io.BytesIO(base64.b64decode(model_file)), map_location=torch.device('cpu')))
        self.sb_net.eval()
        
        # self.SetStartDate(2022, 1, 1)
        # self.SetEndDate(2022, 11, 1)
        # self.SetStartDate(2016, 1, 1)
        # self.SetEndDate(2017, 1, 1)
        # self.SetStartDate(2010, 1, 1)
        # self.SetEndDate(2011, 1, 1)
        self.SetStartDate(2017, 1, 1)
        self.SetEndDate(2021, 1, 1)
        self.SetCash(10**7)
        # self.AddEquity('SPY')
        
        self.numdays = 1000       # length of formation period which determine the copula we use
        self.lookbackdays = 250   # length of history data in trading period
        self.cap_CL = 0.95        # cap confidence level
        self.floor_CL = 0.05      # floor confidence level
        self.weight_v = 0.35       # desired holding weight of asset v in the portfolio, adjusted to avoid insufficient buying power
        self.coef = 0             # to be calculated: requested ratio of quantity_u / quantity_v
        self.window = {}          # stores historical price used to calculate trading day's stock return
        
        self.day = 0              # keep track of current day for daily rebalance
        self.month = 0            # keep track of current month for monthly recalculation of optimal trading pair
        self.pair = []            # stores the selected trading pair
        
        # Select optimal trading pair into the universe
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse('PairUniverse', self.PairSelection)


    def OnData(self, slice):
        '''Main event handler. Implement trading logic.'''
        # self.SetWarmUp(100)

        if self.get_structural_break_prediction() > 0.9:
            return

        self.SetSignal(slice)     # only executed at first day of each month
        # if slice.Bars.ContainsKey('SPY):
        # self.SetHoldings('SPY', 0.4)


        # Daily rebalance
        if self.Time.day == self.day:
            return
        
        long, short = self.pair[0], self.pair[1]

        # Update current price to trading pair's historical price series
        for kvp in self.Securities:
            symbol = kvp.Key
            if symbol in self.pair:
                price = kvp.Value.Price
                self.window[symbol].append(price)

        if len(self.window[long]) < 2 or len(self.window[short]) < 2:
            return
        
        # Compute the mispricing indices for u and v by using estimated copula
        MI_u_v, MI_v_u = self._misprice_index()

        # Placing orders: if long is relatively underpriced, buy the pair
        if MI_u_v < self.floor_CL and MI_v_u > self.cap_CL:
            
            self.SetHoldings(short, -self.weight_v, False, f'Coef: {self.coef}')
            self.SetHoldings(long, self.weight_v * self.coef * self.Portfolio[long].Price / self.Portfolio[short].Price)

        # Placing orders: if short is relatively underpriced, sell the pair
        elif MI_u_v > self.cap_CL and MI_v_u < self.floor_CL:

            self.SetHoldings(short, self.weight_v, False, f'Coef: {self.coef}')
            self.SetHoldings(long, -self.weight_v * self.coef * self.Portfolio[long].Price / self.Portfolio[short].Price)
        
        self.day = self.Time.day


    def SetSignal(self, slice):
        '''Computes the mispricing indices to generate the trading signals.
        It's called on first day of each month'''

        if self.Time.month == self.month:
            return
        
        ## Compute the best copula
        
        # Pull historical log returns used to determine copula
        logreturns = self._get_historical_returns(self.pair, self.numdays)
        x, y = logreturns[str(self.pair[0])], logreturns[str(self.pair[1])]

        # Convert the two returns series to two uniform values u and v using the empirical distribution functions
        ecdf_x, ecdf_y  = ECDF(x), ECDF(y)
        u, v = [ecdf_x(a) for a in x], [ecdf_y(a) for a in y]
        
        # Compute the Akaike Information Criterion (AIC) for different copulas and choose copula with minimum AIC
        tau = kendalltau(x, y)[0]  # estimate Kendall'rank correlation
        AIC ={}  # generate a dict with key being the copula family, value = [theta, AIC]
        
        for i in ['clayton', 'frank', 'gumbel']:
            param = self._parameter(i, tau)
            lpdf = [self._lpdf_copula(i, param, x, y) for (x, y) in zip(u, v)]
            # Replace nan with zero and inf with finite numbers in lpdf list
            lpdf = np.nan_to_num(lpdf) 
            loglikelihood = sum(lpdf)
            AIC[i] = [param, -2 * loglikelihood + 2]
            
        # Choose the copula with the minimum AIC
        self.copula = min(AIC.items(), key = lambda x: x[1][1])[0]
        
        ## Compute the signals
        
        # Generate the log return series of the selected trading pair
        logreturns = logreturns.tail(self.lookbackdays)
        x, y = logreturns[str(self.pair[0])], logreturns[str(self.pair[1])]
        
        # Estimate Kendall'rank correlation
        tau = kendalltau(x, y)[0] 
        
        # Estimate the copula parameter: theta
        self.theta = self._parameter(self.copula, tau)
        
        # Simulate the empirical distribution function for returns of selected trading pair
        self.ecdf_x, self.ecdf_y  = ECDF(x), ECDF(y) 
        
        # Run linear regression over the two history return series and return the desired trading size ratio
        self.coef = stats.linregress(x,y).slope
        
        self.month = self.Time.month
        

    def PairSelection(self, date):
        '''Selects the pair of stocks with the maximum Kendall tau value.
        It's called on first day of each month'''
        
        if date.month == self.month:
            return Universe.Unchanged
        
        symbols = [ Symbol.Create(x, SecurityType.Equity, Market.USA) 
                    for x in [  
                                "EPR", "TGH",
                                "TRTN", "TGH",
                                "TRTN", "SPG",
                                "EPR", "SPG",
                                "SPG", "TGH",
                                "EPR", "TRTN",
                                "SPG", "EPR"
            
                            ] ]

        logreturns = self._get_historical_returns(symbols, self.lookbackdays)
        
        tau = 0
        for i in range(0, len(symbols), 2):
            try:
            
                x = logreturns[str(symbols[i])]
                y = logreturns[str(symbols[i+1])]
                
                # Estimate Kendall rank correlation for each pair
                tau_ = kendalltau(x, y)[0]
                
                if tau > tau_:
                    continue

                tau = tau_
                self.pair = symbols[i:i+2]
            except:
                continue
        
        return [x.Value for x in self.pair]


    def OnSecuritiesChanged(self, changes):
        '''Warms up the historical price for the newly selected pair.
        It's called when current security universe changes'''
        
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            self.window.pop(symbol)
            if security.Invested:
                self.Liquidate(symbol, "Removed from Universe")
        
        for security in changes.AddedSecurities:
            self.window[security.Symbol] = deque(maxlen = 2)
        
        # Get historical prices
        history = self.History(list(self.window.keys()), 2, Resolution.Daily)
        history = history.close.unstack(level=0)
        for symbol in self.window:
            self.window[symbol].append(history[str(symbol)][0])

        
    def _get_historical_returns(self, symbols, period):
        '''Get historical returns for a given set of symbols and a given period
        '''
        
        history = self.History(symbols, period, Resolution.Daily)
        history = history.close.unstack(level=0)
        return (np.log(history) - np.log(history.shift(1))).dropna()
        
        
    def _parameter(self, family, tau):
        ''' Estimate the parameters for three kinds of Archimedean copulas
        according to association between Archimedean copulas and the Kendall rank correlation measure
        '''
        
        if  family == 'clayton':
            return 2 * tau / (1 - tau)
        
        elif family == 'frank':
            
            '''
            debye = quad(integrand, sys.float_info.epsilon, theta)[0]/theta  is first order Debye function
            frank_fun is the squared difference
            Minimize the frank_fun would give the parameter theta for the frank copula 
            ''' 
            
            integrand = lambda t: t / (np.exp(t) - 1)  # generate the integrand
            frank_fun = lambda theta: ((tau - 1) / 4.0  - (quad(integrand, sys.float_info.epsilon, theta)[0] / theta - 1) / theta) ** 2
            
            return minimize(frank_fun, 4, method='BFGS', tol=1e-5).x 
        
        elif family == 'gumbel':
            return 1 / (1 - tau)
            

    def _lpdf_copula(self, family, theta, u, v):
        '''Estimate the log probability density function of three kinds of Archimedean copulas
        '''
        
        if  family == 'clayton':
            pdf = (theta + 1) * ((u * (-theta) + v * (-theta) - 1) * (-2 - 1 / theta)) * (u * (-theta - 1) * v ** (-theta - 1))
            
        elif family == 'frank':
            num = -theta * (np.exp(-theta) - 1) * (np.exp(-theta * (u + v)))
            denom = ((np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1) + (np.exp(-theta) - 1)) ** 2
            pdf = num / denom
            
        elif family == 'gumbel':
            A = (-np.log(u)) * theta + (-np.log(v)) * theta
            c = np.exp(-A ** (1 / theta))
            pdf = c * (u * v) * (-1) * (A * (-2 + 2 / theta)) * ((np.log(u) * np.log(v)) * (theta - 1)) * (1 + (theta - 1) * A * (-1 / theta))
            
        return np.log(pdf)


    def _misprice_index(self):
        '''Calculate mispricing index for every day in the trading period by using estimated copula
        Mispricing indices are the conditional probability P(U < u | V = v) and P(V < v | U = u)'''
        
        return_x = np.log(self.window[self.pair[0]][-1] / self.window[self.pair[0]][-2])
        return_y = np.log(self.window[self.pair[1]][-1] / self.window[self.pair[1]][-2])
        
        # Convert the two returns to uniform values u and v using the empirical distribution functions
        u = self.ecdf_x(return_x)
        v = self.ecdf_y(return_y)
        
        if self.copula == 'clayton':
            MI_u_v = v * (-self.theta - 1) * (u * (-self.theta) + v * (-self.theta) - 1) * (-1 / self.theta - 1) # P(U<u|V=v)
            MI_v_u = u * (-self.theta - 1) * (u * (-self.theta) + v * (-self.theta) - 1) * (-1 / self.theta - 1) # P(V<v|U=u)
    
        elif self.copula == 'frank':
            A = (np.exp(-self.theta * u) - 1) * (np.exp(-self.theta * v) - 1) + (np.exp(-self.theta * v) - 1)
            B = (np.exp(-self.theta * u) - 1) * (np.exp(-self.theta * v) - 1) + (np.exp(-self.theta * u) - 1)
            C = (np.exp(-self.theta * u) - 1) * (np.exp(-self.theta * v) - 1) + (np.exp(-self.theta) - 1)
            MI_u_v = B / C
            MI_v_u = A / C
        
        elif self.copula == 'gumbel':
            A = (-np.log(u)) * self.theta + (-np.log(v)) * self.theta
            C_uv = np.exp(-A ** (1 / self.theta))   # C_uv is gumbel copula function C(u,v)
            MI_u_v = C_uv * (A * ((1 - self.theta) / self.theta)) * (-np.log(v)) * (self.theta - 1) * (1.0 / v)
            MI_v_u = C_uv * (A * ((1 - self.theta) / self.theta)) * (-np.log(u)) * (self.theta - 1) * (1.0 / u)
            
        return MI_u_v, MI_v_u

    def fetch_news(self, page_token):
        stock_1, stock_2 = [stock.Value for stock in self.pair]
        today = date.today()
        start = today - timedelta(days=15)
        url = 'https://data.alpaca.markets/v1beta1/news'
        params={'symbols': ','.join([stock_1, stock_2]),
                'start': start.isoformat(),
                'limit': 50,
                'page_token': page_token}
        req = PreparedRequest()
        req.prepare_url(url, params)
        news_response = self.Download(req.url,
                                    headers={'Apca-Api-Key-Id': self.alpaca_key_id, 'Apca-Api-Secret-Key': self.alpaca_secret_key})
        news_response = json.loads(news_response)
        return news_response

    def add_news_sentiment(self, df):
        stock_1, stock_2 = [stock.Value for stock in self.pair]
        self.Log(stock_1)
        self.Log(stock_2)
        vader = SentimentIntensityAnalyzer()
        next_page_token, first_run = None, True
        while first_run or next_page_token:
            first_run = False
            news_response = self.fetch_news(next_page_token)
            for response in news_response['news']:
                headline = response.get('headline', '')
                summary = response.get('summary', '')
                news_text = f'{headline}, {summary}'
                start_time = dateutil.parser.isoparse(response['created_at'])
                end_time = start_time + timedelta(weeks=1)
                columns = []
                if stock_1 in response['symbols']:
                    columns.append('news_1')
                if stock_2 in response['symbols']:
                    columns.append('news_2')
                df.loc[start_time:, columns] += f' {news_text}'
            next_page_token = news_response['next_page_token']
        df['sentiment_spread'] = df[f'news_1'].apply(lambda news: vader.polarity_scores(news)['compound']) - df[f'news_2'].apply(lambda news: vader.polarity_scores(news)['compound'])
        return df

    def get_structural_break_prediction(self):
        # We buffer in case of missing data to get next best prediction
        stock_1, stock_2 = [stock.Value for stock in self.pair]
        history = self.History(self.pair, 30, Resolution.Daily)
        history = history.unstack(level=0)
        df = pd.DataFrame()
        df['spread'] = np.log(history['close'].iloc[:, 0]) - np.log(history['close'].iloc[:, 1])
        df['avg'] = df['spread'].rolling(window=timedelta(days=90)).mean()
        df['std'] = df['spread'].rolling(window=timedelta(days=90)).std()
        df = df.dropna()
        df['z_score'] = (df['spread'] - df['avg']) / df['std']
        df['volume_1'] = history['volume'].iloc[:, 0]
        df['volume_2'] = history['volume'].iloc[:, 1]
        df['news_1'] = ''
        df['news_2'] = ''
        try:
            df = self.add_news_sentiment(df)
        except:
            # Sometimes due to various errors, we cannot fetch news data. In this
            # case, just don't use news sentiment.
            df['sentiment_spread'] = 0
        df.dropna(inplace=True)
        self.Debug(df.shape)
        df = df.iloc[-15:,:]
        scaler = preprocessing.MinMaxScaler()
        columns = ['spread', 'z_score', 'volume_1', 'volume_2', 'sentiment_spread']
        normalized_df = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns, index=df.index)
        scalogram = torch.tensor(signal.cwt(np.array(df['spread']), signal.ricker, np.arange(1, 500))).unsqueeze(0)
        cnn_X = scalogram[:, 0:9]
        lstm_X = torch.from_numpy(normalized_df.loc[:, ['z_score', 'volume_1', 'volume_2', 'sentiment_spread']].to_numpy())
        with torch.no_grad():
            output = self.sb_net(scalogram, lstm_X.float())
        return output
