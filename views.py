from flask import render_template
from crypto_picker import app
from flask import request
from crypto_picker.a_Model import ModelIt

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import pickle
import numpy as np
import statsmodels.tsa.stattools as ts
import random

# Python code to connect to Postgres
# You may need to modify this based on your OS, 
# as detailed in the postgres dev setup materials.
# user = 'jcbrooks' #add your Postgres username here      
# host = 'localhost'
# dbname = 'test_crypto_data'
# db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)

# user1 = 'jared' #add your Postgres username here      
# host1 = 'insight-prod-cluster.cdu1ardlv7nm.us-east-1.redshift.amazonaws.com'
# dbname1 = 'prod_insight'
# password1 = 'VmFmKvKZEW7aL3oQnqmJAiZZDRTUJ7D'
# port1 = 5439
# con1 = None
# con1 = psycopg2.connect(database = dbname1, user = user1, password=password1, host=host1, port=port1)

lr_clf = pickle.load(open('crypto_picker/lr_model.sav', 'rb'))
print(type(lr_clf))

# @app.route('/')
# @app.route('/index')
# def index():
#     return render_template("index.html",
#        title = 'Home', user = { 'nickname': 'Miguel' },
#        )

# @app.route('/db')
# def birth_page():
#     sql_query = """                                                             
#                 SELECT * FROM birth_data_table WHERE delivery_method='Cesarean'\
# ;                                                                               
#                 """
#     query_results = pd.read_sql_query(sql_query,con)
#     births = ""
#     print(query_results[:10])
#     for i in range(0,10):
#         births += query_results.iloc[i]['birth_month']
#         births += "<br>"
#     return births

# @app.route('/db_fancy')
# def cesareans_page_fancy():
#     sql_query = """
#                SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean';
#                 """
#     query_results=pd.read_sql_query(sql_query,con)
#     births = []
#     for i in range(0,query_results.shape[0]):
#         births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
#     return render_template('cesareans.html',births=births)

# @app.route('/input')

@app.route('/')
@app.route('/index')
def crypto_input():
  # query = "SELECT * FROM market_cap_table;"
  # print(query)
  # query_results=pd.read_sql_query(query,con)
  # print(query_results['symbol'])
  market_cap_table = pd.read_csv('crypto_picker/market_cap_table.csv')
  curr = market_cap_table['symbol']
  curr = list(curr)
  curr.insert(0,'')
  # for c in curr:
  #   print(c)
  return render_template("input.html", curr = curr)

# @app.route('/output')
# def cesareans_output():
#     return render_template("output.html")

@app.route('/output')
def crypto_output():

  rand = request.args.get('random')
  big = request.args.get('big')
  small = request.args.get('small')
  # print(random)

  crypto1 = request.args.get('crypto1')
  value1 = request.args.get('value1')

  crypto2 = request.args.get('crypto2')
  value2 = request.args.get('value2')

  crypto3 = request.args.get('crypto3')
  value3 = request.args.get('value3')

  crypto4 = request.args.get('crypto4')
  value4 = request.args.get('value4')

  crypto5 = request.args.get('crypto5')
  value5 = request.args.get('value5')

  crypto6 = request.args.get('crypto6')
  value6 = request.args.get('value6')

  # query1 = "SELECT * FROM curr_info_table;"
  # curr_info=pd.read_sql_query(query1,con)
  curr_info = pd.read_csv('crypto_picker/curr_info_table.csv')
  print(curr_info)
  curr = curr_info['curr']
  curr = list(curr)

  cryptos = [crypto1, crypto2, crypto3, crypto4, crypto5, crypto6]
  weights = [value1, value2, value3, value4, value5, value6]

  cryptos = list(filter(None, cryptos))
  weights = list(filter(None, weights))

  weights = [float(i) for i in weights]
  weight_sum = sum(weights)
  norm_weights = []
  for w in weights:
    norm_weights.append(w/weight_sum)

  df_market_proportions = curr_info[['curr', 'mc']].copy()
  total_mc = sum(df_market_proportions['mc'])
  df_market_proportions['mc_frac'] = df_market_proportions['mc']/total_mc
  curr_list_proportional = []

  for index, row in df_market_proportions.iterrows():
    num_entries = int(1000*row['mc_frac'])
    for i in range(num_entries):
        curr_list_proportional.append(row['curr'])

  if(rand=='Random Portfolio'):
    num_of_cryptos = [3,4,5,6]
    num = random.choice(num_of_cryptos)
    # cryptos = []
    # while len(cryptos)<num:
    #     add = random.choice(curr_list_proportional)
    #     if(add not in cryptos):
    #         cryptos.append(add)
    cryptos = random.sample(curr, num)
    norm_weights = random_weights(num)
    weights = []
    for w in norm_weights:
      weights.append(round(w*np.random.uniform(high=1000, low=20),2))


  if(big=='Big Coin Investor'):
    cryptos = ['BTC', 'ETH', 'XRP', 'BCH', 'LTC']
    norm_weights = [0.5, 0.2, 0.15, 0.05, 0.1]
    weights = []
    for w in norm_weights:
      weights.append(round(w*847.54,2))


  if(small=='Small Coin Investor'):
    cryptos = ['STEEM', 'BCN', 'XVG', 'BTS']
    norm_weights = [0.4, 0.25, 0.2, 0.15]
    weights = []
    for w in norm_weights:
      weights.append(round(w*673.95,2))

  prices = []
  for c in cryptos:
    prices.append(curr_info[curr_info['curr']==c]['last_price'].values[0])

  print(cryptos)
  print(norm_weights)

  portfolio = pd.DataFrame(
    {'cryptos': cryptos,
     'weights': weights,
     'norm_weights': norm_weights,
     'prices': prices
    })  
    #just select the Cesareans  from the birth dtabase for the month that the user inputs

  # query = "SELECT * FROM daily_return_table;"
  #query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
  #print(query)
  # query_results=pd.read_sql_query(query,con)
  #print(query_results)

  daily_return_table = pd.read_csv('crypto_picker/daily_return_table.csv')

  sug_results = suggest_curr(cryptos, norm_weights, curr, curr_info, daily_return_table)
  print(sug_results['curr'])

  prices_sugg = []
  day_change = []
  week_change = []
  month_change = []
  for c in sug_results['curr']:
    prices_sugg.append(curr_info[curr_info['curr']==c]['last_price'].values[0])
    day_change.append(round(curr_info[curr_info['curr']==c]['day_change'].values[0],1))
    week_change.append(round(curr_info[curr_info['curr']==c]['week_change'].values[0],1))
    month_change.append(round(curr_info[curr_info['curr']==c]['month_change'].values[0],1))

  print(portfolio)

  portfolio_list = []

  for i in range(0,portfolio.shape[0]):
    portfolio_list.append(dict(cryptos=portfolio.iloc[i]['cryptos'], prices=portfolio.iloc[i]['prices'], weights=portfolio.iloc[i]['weights'], norm_weights=portfolio.iloc[i]['norm_weights']))

  suggs = []
  for i in range(0,sug_results.shape[0]):
      suggs.append(dict(curr=sug_results.iloc[i]['curr'], probs=sug_results.iloc[i]['probs'], price=prices_sugg[i], day_change=day_change[i], week_change=week_change[i], month_change=month_change[i], addition_marketcap=round((sug_results.iloc[i]['addition_marketcap']/1e6),1)))
  #     the_result = ''
  # the_result = ModelIt(patient,births)

  message1 = ''
  message2 = ''
  if(len(suggs)==0):
    message1='Your portfolio is balanced and performing well! '
    message2='There are no suggestions for you at this time.'


  return render_template("output.html", portfolio_list = portfolio_list, suggs=suggs, message1=message1, message2=message2)
  #return render_template("output.html", births = births, the_result = the_result)


def er_and_risk(p, num, weights, df):
    p_er = 0
    p_risk = 0
    #print(num, len(p))
    
    for j in range(num):
        p_er += (weights[j])*np.mean(df[p[j]])
        p_risk += (weights[j])**2*np.var(df[p[j]])
        for k in range(num):
            if(j!=k):
#                 print(len(df[p[k]]))
#                 print(df[p[j]].corr(df[p[k]]))
                p_risk += (weights[j])*(weights[k])*np.std(df[p[j]])*np.std(df[p[k]])*df[p[j]].corr(df[p[k]])
    p_risk = np.sqrt(p_risk)
    
    return p_er, p_risk


scalerfile = 'crypto_picker/scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))

thresh = 0.28

#def suggest_curr(portfolio, weights, clf, top_curr, market_caps, df2_train_period):
def suggest_curr(portfolio, weights, top_curr, curr_info, df2_train_period):
    initial_er = []
    initial_risk = []
    newp_er = []
    newp_risk = []
    delta_er = []
    delta_risk = []
    correlation = []
    cointegration = []
    addition_er = []
    addition_performance = []
    addition_std = []
    addition_marketcap = []
    curr = []
    
    ini_er, ini_risk = er_and_risk(portfolio, len(portfolio), weights, df2_train_period)
    for c in top_curr:
        if(c in portfolio or c=='BTC'):
            continue
        
        new_portfolio = portfolio.copy()
        new_portfolio.append(c)
        new_weights = []
        for w in weights:
            new_weights.append(w*(1/1.2))
        new_weights.append(1-sum(new_weights))

        num = len(new_portfolio)
        new_er, new_risk = er_and_risk(new_portfolio, num, new_weights, df2_train_period)

        curr.append(c)
        
        initial_er.append(ini_er)
        initial_risk.append(ini_risk)
        newp_er.append(new_er)
        newp_risk.append(new_risk)
        delta_er.append(new_er-ini_er)
        delta_risk.append(new_risk-ini_risk)
        
        addition_er.append(np.mean(df2_train_period[c]))
        addition_std.append(np.std(df2_train_period[c]))
        
        portfolio_performance = pd.Series([])
        for k in range(len(weights)):
            new = weights[k]*df2_train_period[portfolio[k]]
            portfolio_performance = portfolio_performance.add(new,fill_value=0)
        
        corr = portfolio_performance.corr(df2_train_period[c])
        coin = ts.coint(portfolio_performance, df2_train_period[c])
        correlation.append(corr)
        cointegration.append(coin[0])
        
        #mc = df_train_period[df_train_period['Symbol']==c]['Market Cap'].iloc[-1]
        mc = curr_info[curr_info['curr']==c]['mc'].values[0]
        addition_marketcap.append(mc)
        
    features = pd.DataFrame(
        {'curr': curr,
         'initial_er': initial_er,
         'initial_risk':initial_risk,
         'newp_er': newp_er,
         'newp_risk': newp_risk,
         'delta_er': delta_er,
         'delta_risk': delta_risk,
         'addition_er': addition_er,
         'addition_std': addition_std,
         'correlation': correlation,
         'cointegration': cointegration,
         'addition_marketcap': addition_marketcap,
           })
    # X = features[['initial_er', 'initial_risk', 'delta_er', 'delta_risk', 
    #          'addition_er', 'addition_std', 'correlation', 'cointegration', 'addition_marketcap']]

    X = features[['initial_er', 'initial_risk', 'delta_er', 'delta_risk', 
             'correlation', 'cointegration']]
    
    X_scaled = scaler.transform(X)
       
    probs = pd.DataFrame(lr_clf.predict_proba(X_scaled))
    features['probs'] = probs[1]
    features = features[features['probs']>thresh]
    features['scaled_probs'] = features['probs']-thresh
    features.sort_values(by='probs', ascending=False, inplace=True)
       
    return features[0:5]

def random_weights(n):
    weights = []
    for i in range(n):
        if(i==n-1):
            weights.append(1-sum(weights))
        else:
            high = 1 - sum(weights)
            weights.append(np.random.uniform(high=high))
    return(weights)

