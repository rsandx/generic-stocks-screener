import smtplib, ssl, contextlib, traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from sqlalchemy import create_engine
import tweepy

#mapping of TA functions using Technical Analysis Library in Python (https://github.com/bukosabino/ta) 
#name: function_name, data_series (among open, high, low, close and volume), number of extra parameters
ta_mapping = {'AO':('momentum.ao', 'high', 'low', 2),
            'KAMA':('momentum.kama', 'close', 3),
            'MFI':('momentum.money_flow_index', 'high', 'low', 'close', 'volume', 1),
            'ROC':('momentum.roc', 'close', 1),
            'RSI':('momentum.rsi', 'close', 1),
            'Fast Stochastic':('momentum.stoch', 'high', 'low', 'close', 1),
            'Slow Stochastic':('momentum.stoch_signal', 'high', 'low', 'close', 2),
            'TSI':('momentum.tsi', 'close', 2),
            'UO':('momentum.uo', 'high', 'low', 'close', 6),
            'Williams %R':('momentum.wr', 'high', 'low', 'close', 1),
            'ADI':('volume.acc_dist_index', 'high', 'low', 'close', 'volume', 0),
            'CMF':('volume.chaikin_money_flow', 'high', 'low', 'close', 'volume', 1),
            'EMV':('volume.ease_of_movement', 'high', 'low', 'close', 'volume', 1),
            'FI':('volume.force_index', 'close', 'volume', 1),
            'NVI':('volume.negative_volume_index', 'close', 'volume', 0),
            'OBV':('volume.on_balance_volume', 'close', 'volume', 0),
            'VPT':('volume.volume_price_trend', 'close', 'volume', 0),
            'ATR':('volatility.average_true_range', 'high', 'low', 'close', 1),
            'Upper Bollinger Band':('volatility.bollinger_hband', 'close', 2),
            'Lower Bollinger Band':('volatility.bollinger_lband', 'close', 2),
            'Median Bollinger Band':('volatility.bollinger_mavg', 'close', 2),    #the function needs just 1 parameter, use 2 as per convention
            'Width of Bollinger Band':('volatility.bollinger_wband', 'close', 2),
            'Percentage of Bollinger Band':('volatility.bollinger_pband', 'close', 2),
            'Upper Donchian Channel':('volatility.donchian_channel_hband', 'high', 'low', 'close', 1),
            'Lower Donchian Channel':('volatility.donchian_channel_lband', 'high', 'low', 'close', 1),
            'Median Donchian Channel':('volatility.donchian_channel_mband', 'high', 'low', 'close', 1),
            'Width of Donchian Channel':('volatility.donchian_channel_wband', 'high', 'low', 'close', 1),
            'Percentage of Donchian Channel':('volatility.donchian_channel_pband', 'high', 'low', 'close', 1),
            'Upper Keltner Channel':('volatility.keltner_channel_hband', 'high', 'low', 'close', 1),
            'Lower Keltner Channel':('volatility.keltner_channel_lband', 'high', 'low', 'close', 1),
            'Median Keltner Channel':('volatility.keltner_channel_mband', 'high', 'low', 'close', 1),
            'Width of Keltner Channel':('volatility.keltner_channel_wband', 'high', 'low', 'close', 1),
            'Percentage of Keltner Channel':('volatility.keltner_channel_pband', 'high', 'low', 'close', 1),
            'ADX':('trend.adx', 'high', 'low', 'close', 1),
            '+DI':('trend.adx_pos', 'high', 'low', 'close', 1),
            '-DI':('trend.adx_neg', 'high', 'low', 'close', 1),
            'CCI':('trend.cci', 'high', 'low', 'close', 1),
            'Aroon Up':('trend.aroon_up', 'close', 1),
            'Aroon Down':('trend.aroon_down', 'close', 1),
            'DPO':('trend.dpo', 'close', 1),
            'EMA':('trend.ema_indicator', 'close', 1),
            'KST':('trend.kst', 'close', 8),
            'MACD':('trend.macd', 'close', 3),    #the function needs just 2 parameters, use 3 as per convention 
            'MACD Signal':('trend.macd_signal', 'close', 3),
            'MACD Histogram':('trend.macd_diff', 'close', 3),
            'TRIX':('trend.trix', 'close', 1),
            'Mass Index':('trend.mass_index', 'high', 'low', 2),
            'Ichimoku A':('trend.ichimoku_a', 'high', 'low', 3),
            'Ichimoku B':('trend.ichimoku_b', 'high', 'low', 3),
            'Ichimoku Base Line':('trend.ichimoku_base_line', 'high', 'low', 3),
            'Ichimoku Conversion Line':('trend.ichimoku_conversion_line', 'high', 'low', 3),
            '+VI':('trend.vortex_indicator_pos', 'high', 'low', 'close', 1),
            '-VI':('trend.vortex_indicator_neg', 'high', 'low', 'close', 1),
            'PSAR Up':('trend.psar_up', 'high', 'low', 'close', 1),
            'PSAR Down':('trend.psar_down', 'high', 'low', 'close', 1),
            'PSAR Up Signal':( 'trend.psar_up_indicator', 'high', 'low', 'close', 1),
            'PSAR Down Signal':('trend.psar_down_indicator', 'high', 'low', 'close', 1),
            'MA':('trend.sma_indicator', 'close', 1),
            'SMA':('trend.sma_indicator', 'close', 1),
            'Volume MA':('trend.sma_indicator', 'volume', 1)
        }

date_format = '%Y-%m-%d'
regions = {1:'APAC', 2:'EMEA', 3:'Americas'}

#create sqlalchemy database connection pool using fast mysqlclient, it can be safely used for multiprocessing with the dispose method
engine = create_engine("mysql+mysqldb://mydb_user:mydb_pwd@localhost/mydb")

#mapping of candlestick patterns using Python wrapper for TA-Lib (https://mrjbq7.github.io/ta-lib/)
#name: (function_name, signal (1: bullish, -1: bearish, 0: neutral), performance rank)
cp_mapping = None
def get_cp_mapping():
    # declare cp_mapping as global so that the function has (write) access to the global var.
    global cp_mapping
    if cp_mapping is None:
        cp_mapping = {}
        query = "SELECT name, functionName, sign, performanceRank FROM candlestickpattern order by performanceRank"
        with contextlib.closing(engine.raw_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            screeners = []
        for r in rows:
            cp_mapping[r[0]] = (r[1], r[2], r[3])
    return cp_mapping

# email handling
mail_signature = ' \n\nHappy screening! \n\nThe World Stocks Screener team'

def send(receiver, email, logger):
    # Log in to email account.
    sender = os.getenv('EMAIL_ADDRESS')
    email["From"] = sender
    try:
        with smtplib.SMTP(os.getenv('EMAIL_SMTP_SERVER'), os.getenv('EMAIL_SMTP_PORT')) as smtpObj:
            #smtpObj.set_debuglevel(1)
            smtpObj.starttls()
            smtpObj.ehlo()
            smtpObj.login(os.getenv('EMAIL_ADDRESS'), os.getenv('EMAIL_PASSWORD'))
            smtpObj.sendmail(sender, receiver, email.as_string())
            logger.debug(f'Email sent to {receiver}')
    except Exception as e:
        logger.error(f'There was a problem sending email to {receiver}: {e}')
        #print(traceback.format_exc())
    
def contains_non_ascii_characters(str):
    return not all(ord(c) < 128 for c in str)   

def add_header(message, header_name, header_value):
    if contains_non_ascii_characters(header_value):
        h = Header(header_value, 'utf-8')
        message[header_name] = h
    else:
        message[header_name] = header_value    
    return message

def sendMail(receiver, subject, body, logger):
    email = MIMEMultipart("alternative")
    email = add_header(email, 'Subject', subject)
    email['To'] = receiver
    
    # Create the plain-text and HTML version of your message
    text = body
    html = """\
<html>
  <body>
    <p>Hi,<br>
       How are you?<br>
       <a href="https://www.aifinancials.net">AI Financials</a> 
       at your service!
    </p>
  </body>
</html>
"""

    if(contains_non_ascii_characters(text)):
        plain_text = MIMEText(text.encode('utf-8'), 'plain', 'utf-8') 
    else:
        plain_text = MIMEText(text, 'plain')
    email.attach(plain_text)

    #if contains_non_ascii_characters(html):
    #    html_text = MIMEText(html.encode('utf-8'), 'html', 'utf-8')
    #else:
    #    html_text = MIMEText(html, 'html')    
    #email.attach(html_text)
    #logger.info(email)
    send(receiver, email, logger)


# twitter handling
def create_api(logger):
    consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
    consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
    access_token = os.getenv('TWITTER_ACCESS_TOKEN')
    access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    try:
        api.verify_credentials()
    except Exception as e:
        logger.error("Error creating API", exc_info=True)
        raise e
    #logger.info("API created")
    return api

def postTweet(subject, logger, messages=None):
    try:
        api = create_api(logger)
        tweet = api.update_status(status = subject) 
        if messages is not None and isinstance(messages, list):
            for message in messages:
                api.update_status(status = message[:140], in_reply_to_status_id = tweet.id, auto_populate_reply_metadata=True)
        #logger.debug(f'posted tweet {subject}')
    except Exception as e:
        logger.error(f'There was a problem posting tweet {subject}: {e}')

def followFollowers(api, logger):
    logger.info("Retrieving and following followers")
    for follower in tweepy.Cursor(api.followers).items():
        if not follower.following:
            logger.info(f"Following {follower.name}")
            follower.follow()

