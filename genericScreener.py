#! python3

import pandas as pd
import numpy as np
import re, json
from ta import *
import sys, os, logging, logging.config, traceback, concurrent_log_handler, getopt
import multiprocessing as mp
import contextlib
import heapq  
import talib
from datetime import datetime, timedelta
#import numba as nb
#from timeit import default_timer as timer

# Internal imports
import utils

logging.config.fileConfig("logging.cfg")
logger = logging.getLogger(os.path.basename(__file__))

#define constants 
IBDRS = 'ibd relative strength'

#group ta names by number of parameters
ta_names = {} 
TA_MAPPING = dict((k.lower(), v) for k,v in utils.ta_mapping.items())
#sort by name in descending order to make sure the longest name is matched
sorted_ta_mapping = {k: v for k, v in sorted(TA_MAPPING.items(), key=lambda x: x[0], reverse=True)}
for k, v in sorted_ta_mapping.items():
    k = k.replace('+', '\+').replace(' ', '\s+')
    if v[-1] not in ta_names:
        ta_names[v[-1]] = k
    else:
        ta_names[v[-1]] += '|' + k
#logger.debug(ta_names)
taRegex = ''
for k, v in ta_names.items():
    if k > 0:
        if '|' in v:
            v = '(' + v + ')'
        v += '\s*\(\s*' + ('\d+\.?\d*,\s*'*k)[:-4] + '\s*\)'
    taRegex += v + '|'
taRegex = taRegex[:-1]
PERIOD = r'(days|weeks|months|day|week|month)'
INDICATOR_PLAIN = r'({}|open|high|low|close|volume|range)'.format(taRegex)
INDICATOR_FUNCTION = r'(min|max|avg)\s*\(\s*({}|open|high|low|close|volume|range),\s*[1-9]\d*\s*\)'.format(taRegex)
INDICATOR = r'((daily|weekly|monthly)\s+)?({0}|{1})(\s+[1-9]\d*\s+{2}\s+ago)?'.format(INDICATOR_PLAIN, INDICATOR_FUNCTION, PERIOD)
#logger.debug(indicator)

CP_NAMES = '|'.join([name.lower().replace(' ', '\s+') for name in sorted(utils.get_cp_mapping().keys(), reverse=True)])
CANDLESTICK_PATTERN = r'((?P<timeframe>daily|weekly|monthly)\s+)?(?P<cspattern>{0})'.format(CP_NAMES)

IS_ABOVE_BELOW_BETWEEN = r'''(?P<indicator>({0}))\s+(is|was|has\s+been|had\s+been)\s+            
    ((?P<more_less>((more|less)\s+than\s+\d+\.?\d*(%|\s+point|\s+points)\s+))?(?P<above_below>above|below)\s+((?P<above_below_indicator>({0}))|(?P<above_below_value>-?\d+\.?\d*))                    
    | ((?P<between>from)\s+((?P<between_indicator1>({0}))|(?P<between_value1>-?\d+\.?\d*))\s+to\s+((?P<between_indicator2>({0}))|(?P<between_value2>-?\d+\.?\d*))))                        
    (\s+for\s+the\s+last\s+(?P<duration>[1-9]\d*\s+{1}))?'''.format(INDICATOR, PERIOD)                     

CROSSED_ABOVE_BELOW = r'''(?P<indicator>({0}))\s+(crossed|has\s+crossed)\s+
    (?P<above_below>above|below)\s+((?P<above_below_indicator>({0}))|(?P<above_below_value>-?\d+\.?\d*))                        
    (\s+within\s+the\s+last\s+(?P<duration>[1-9]\d*\s+{1}))?'''.format(INDICATOR, PERIOD)

DROPPED_GAINED = r'''(?P<indicator>({0}))\s+(?P<verb>dropped|gained)\s+                       
    (?P<more_less>((more|less)\s+than\s+\d+\.?\d*(%|\s+point|\s+points)))\s+                   
    (over\s+the\s+last\s+(?P<duration>[1-9]\d*\s+{1}))?'''.format(INDICATOR, PERIOD)

INCREASING_DECREASING = r'''(?P<indicator>({0}))\s+has\s+been\s+(?P<verb>increasing|decreasing)\s+                       
    (for\s+(?P<duration>[1-9]\d*\s+{1}))'''.format(INDICATOR, PERIOD)

REACHED_HIGH_LOW = r'''(?P<indicator>({0}))\s+(reached|has\s+reached)\s+
    a\s+new\s+(?P<high_low>([1-9]\d*\s+{1}\s+(high|low)))                        
    (\s+within\s+the\s+last\s+(?P<duration>[1-9]\d*\s+{1}))?'''.format(INDICATOR, PERIOD)

TOP_BOTTOM = r'''(?P<top_botom>top|bottom)\s+(?P<number>[1-9]\d*)\s+
    ((?P<indicator>({0}))|IBD\s+Relative\s+Strength)'''.format(INDICATOR)

FORMED = r'''{0}\s+(formed|has\s+formed)(\s+within\s+the\s+last\s+(?P<duration>[1-9]\d*\s+{1}))?'''.format(CANDLESTICK_PATTERN, PERIOD)

indicator_1 = r'((?P<timeframe>daily|weekly|monthly)\s+)?(?P<indicator>{0})(?P<offset>\s+[1-9]\d*\s+{1}\s+ago)?'.format(INDICATOR_PLAIN, PERIOD)
indicator_2 = r'((?P<timeframe>daily|weekly|monthly)\s+)?(min|max|avg)\s*\(\s*(?P<indicator>{0}),\s*(?P<range>[1-9]\d*)\s*\)\s*(?P<offset>\s+[1-9]\d*\s+{1}\s+ago)?'.format(INDICATOR_PLAIN, PERIOD)
PLAIN_INDICATOR_RE = re.compile(indicator_1, re.IGNORECASE | re.VERBOSE)
AGGREGATE_INDICATOR_RE = re.compile(indicator_2, re.IGNORECASE | re.VERBOSE)


def evaluate(a):
    return eval(a)

#eval_nb = nb.njit(evaluate)
#so far numba can't be used as njit numpy calculation can return incorrect result (https://github.com/numba/numba/issues/4419)

def tolist(pandas_series):
    return None if pandas_series is None else pandas_series.fillna(-999999999).round(4).tolist()
    #return None if pandas_series is None else pandas_series.fillna(method='bfill').dropna().round(4).tolist()

class MyScreener:

    def __init__(self):
        self._id = None
        self._expression = None
        self._exchanges = None
        self._symbols = None
        self._priceType = None
        self._volumeType = None
        self._priceLow = None
        self._priceHigh = None
        self._volumeLow = None
        self._volumeHigh = None
        self._translation = None
        self._industries = None

    @property
    def id(self):
        return self._id
        
    @id.setter
    def id(self, value):
        self._id = value
        
    @property
    def expression(self):
        return self._expression
        
    @expression.setter
    def expression(self, value):
        self._expression = value
        
    @property
    def exchanges(self):
        return self._exchanges
        
    @exchanges.setter
    def exchanges(self, value):
        self._exchanges = value
        
    @property
    def symbols(self):
        return self._symbols
        
    @symbols.setter
    def symbols(self, value):
        self._symbols = value
        
    @property
    def priceType(self):
        return self._priceType
        
    @priceType.setter
    def priceType(self, value):
        self._priceType = value
        
    @property
    def volumeType(self):
        return self._volumeType
        
    @volumeType.setter
    def volumeType(self, value):
        self._volumeType = value
        
    @property
    def priceLow(self):
        return self._priceLow
        
    @priceLow.setter
    def priceLow(self, value):
        self._priceLow = value
        
    @property
    def priceHigh(self):
        return self._priceHigh
        
    @priceHigh.setter
    def priceHigh(self, value):
        self._priceHigh = value
        
    @property
    def volumeLow(self):
        return self._volumeLow
        
    @volumeLow.setter
    def volumeLow(self, value):
        self._volumeLow = value
        
    @property
    def volumeHigh(self):
        return self._volumeHigh
        
    @volumeHigh.setter
    def volumeHigh(self, value):
        self._volumeHigh = value
        
    @property
    def translation(self):
        return self._translation
        
    @translation.setter
    def translation(self, value):
        self._translation = value
        
    @property
    def industries(self):
        return self._industries
        
    @industries.setter
    def industries(self, value):
        self._industries = value
        
        
    def __translate(self, statement):
        """Translate a statement to a list of expressions. 
        For example, [0, ["weekly", "+di(13)", 0, 13], ["", "above"], ["weekly", "-di(13)", 0, 13], null].
        The first value of the list is type: 
            1 - 'is above/below', 
            2 - 'is in between', 
            3 - 'crossed above/below'
            4 - 'gained'
            4.1 - 'dropped'
            5 - 'increase'
            5.1 - 'decrease'
            6 - 'reach high/low'
            7 - 'top'
            8 - 'bottom'
            99 - 'form' (candlestick pattern)
        """        
        
        sceenerRegex = re.compile(IS_ABOVE_BELOW_BETWEEN, re.IGNORECASE | re.VERBOSE)
        mo = sceenerRegex.search(statement)
        if mo is not None:
            logger.debug(f"{mo.group('indicator')}, {mo.group('more_less')}, {mo.group('above_below')}, {mo.group('above_below_indicator')}, {mo.group('above_below_value')}, \
{mo.group('between')}, {mo.group('between_indicator1')}, {mo.group('between_value1')}, {mo.group('between_indicator2')}, {mo.group('between_value2')}, {mo.group('duration')}")
            indicator = mo.group('indicator').strip().lower()
            value1 = getIndicatorComponents(indicator)
            duration = mo.group('duration')
            if duration is not None:
                duration = getOffset(duration.strip(), value1[0])
                value1[-1] += duration

            if mo.group('above_below') is not None:
                type = 1
                comparator = [None if mo.group('more_less') is None else mo.group('more_less').strip(), mo.group('above_below')]
                value2 = mo.group('above_below_value')
                if value2 is not None:
                    if duration is not None or statement.rstrip().endswith(value2):
                        return [type, value1, comparator, value2, duration]
                value2 = mo.group('above_below_indicator')
                if value2 is not None:
                    value2 = getIndicatorComponents(value2.strip().lower())
                    if duration is not None:
                        if value1[0] == value2[0]:
                            value2[-1] += duration
                        else:
                            value2[-1] += getOffset(mo.group('duration'), value2[0])
                    if duration is not None or statement.rstrip().endswith(mo.group('above_below_indicator')):
                        return [type, value1, comparator, value2, duration]

            if mo.group('between') is not None:
                type = 2
                between_value1 = mo.group('between_value1')
                between_indicator1 = mo.group('between_indicator1')
                if between_indicator1 is not None:
                    between_indicator1 = mo.group('between_indicator1').strip().lower()
                    between_value1 = getIndicatorComponents(between_indicator1)
                    if duration is not None:
                        if value1[0] == between_value1[0]:
                            between_value1[-1] += duration
                        else:
                            between_value1[-1] += getOffset(mo.group('duration'), between_value1[0])
                between_value2 = mo.group('between_value2')
                if between_value2 is not None:
                    noduration = statement.rstrip().endswith(between_value2)
                between_indicator2 = mo.group('between_indicator2')
                if between_indicator2 is not None:
                    noduration = statement.rstrip().endswith(between_indicator2)
                    between_value2 = getIndicatorComponents(between_indicator2.strip().lower())
                    if duration is not None:
                        if value1[0] == between_value2[0]:
                            between_value2[-1] += duration
                        else:
                            between_value2[-1] += getOffset(mo.group('duration'), between_value2[0])
                if between_value1 is not None and between_value2 is not None:
                    if duration is not None or noduration:
                        return [type, value1, between_value1, between_value2, duration]

        sceenerRegex = re.compile(CROSSED_ABOVE_BELOW, re.IGNORECASE | re.VERBOSE)
        mo = sceenerRegex.search(statement)
        if mo is not None:
            type = 3
            logger.debug(f"{mo.group('indicator')}, {mo.group('above_below')}, {mo.group('above_below_indicator')}, {mo.group('above_below_value')}, {mo.group('duration')}")
            indicator = mo.group('indicator').strip().lower()
            value1 = getIndicatorComponents(indicator)
            duration = mo.group('duration')
            if duration is not None:
                duration = getOffset(duration.strip(), value1[0])
                value1[-1] += duration
            comparator = mo.group('above_below')
            value2 = mo.group('above_below_value')
            if value2 is not None:
                if duration is not None or statement.rstrip().endswith(value2):
                    return [type, value1, comparator, value2, duration]
            value2 = mo.group('above_below_indicator')
            if value2 is not None:
                value2 = getIndicatorComponents(value2.strip().lower())
                if duration is not None:
                    if value1[0] == value2[0]:
                        value2[-1] += duration
                    else:
                        value2[-1] += getOffset(mo.group('duration'), value2[0])
                if duration is not None or statement.rstrip().endswith(mo.group('above_below_indicator')):
                    return [type, value1, comparator, value2, duration]

        sceenerRegex = re.compile(DROPPED_GAINED, re.IGNORECASE | re.VERBOSE)
        mo = sceenerRegex.search(statement)
        if mo is not None:
            logger.debug(f"{mo.group('indicator')}, {mo.group('verb')}, {mo.group('more_less')}, {mo.group('duration')}")
            if mo.group('verb').lower() == 'gained':
                type = 4
            else:
                type = 4.1
            indicator = mo.group('indicator').strip().lower()
            value1 = getIndicatorComponents(indicator)
            duration = mo.group('duration')
            if duration is not None:
                duration = getOffset(duration.strip(), value1[0])
                value1[-1] += duration
            comparator = mo.group('more_less')
            if duration is not None or statement.rstrip().endswith(comparator):
                return [type, value1, comparator, duration]
    
        sceenerRegex = re.compile(INCREASING_DECREASING, re.IGNORECASE | re.VERBOSE)
        mo = sceenerRegex.search(statement)
        if mo is not None:
            logger.debug(f"{mo.group('indicator')}, {mo.group('verb')}, {mo.group('duration')}")
            if mo.group('verb').lower() == 'increasing':
                type = 5
            else:
                type = 5.1
            indicator = mo.group('indicator').strip().lower()
            value1 = getIndicatorComponents(indicator)
            duration = mo.group('duration')
            duration = getOffset(duration.strip(), value1[0])
            value1[-1] += duration
            return [type, value1, duration]
    
        sceenerRegex = re.compile(REACHED_HIGH_LOW, re.IGNORECASE | re.VERBOSE)
        mo = sceenerRegex.search(statement)
        if mo is not None:
            type = 6
            logger.debug(f"{mo.group('indicator')}, {mo.group('high_low')}, {mo.group('duration')}")
            indicator = mo.group('indicator').strip().lower()
            value1 = getIndicatorComponents(indicator)
            duration = mo.group('duration')
            if duration is not None:
                duration = getOffset(duration.strip(), value1[0])
                value1[-1] += duration
            value2 = mo.group('high_low')
            if duration is not None or statement.rstrip().endswith(value2):
                return [type, value1, value2, duration]

        sceenerRegex = re.compile(TOP_BOTTOM, re.IGNORECASE | re.VERBOSE)
        mo = sceenerRegex.search(statement)
        if mo is not None:
            logger.debug(f"{mo.group('top_botom')}, {mo.group('number')}, {mo.group('indicator')}")
            if mo.group('top_botom').lower() == 'top':
                type = 7
            else:
                type = 8
            if mo.group('indicator') is None:
                value1 = IBDRS
            else:
                indicator = mo.group('indicator').strip().lower()
                value1 = getIndicatorComponents(indicator)
            return [type, mo.group('number'), value1]

        sceenerRegex = re.compile(FORMED, re.IGNORECASE | re.VERBOSE)
        mo = sceenerRegex.search(statement)
        if mo is not None:
            type = 99
            logger.debug(f"{mo.group('timeframe')}, {mo.group('cspattern')}, {mo.group('duration')}")
            timeframe = mo.group('timeframe')
            if timeframe is None: 
                timeframe = 'daily'
            else:
                timeframe = timeframe.lower()
            cspattern = mo.group('cspattern').strip().lower()
            cspattern = ' '.join(cspattern.split())   #remove extra whitespace  
            duration = mo.group('duration')
            if duration is not None:
                duration = getOffset(duration.strip(), timeframe)
            if duration is not None or statement.rstrip().lower().endswith('formed'):
                return [type, timeframe, cspattern, duration]
            
        errorMessage = f'"{statement}" is unrecognizable, please check against the acceptable syntax, make sure the candlestick pattern name or indicator name is spelled correctly, and required parameters are included.'
        # (http://screenerapp.aifinancials.net/screenerSyntax)
        logger.error(errorMessage)
        raise Exception(errorMessage)
        

    def __separate(self, expression):
        expression = expression.replace('\n', ' ').strip()
        if ('top' in expression.lower() or 'bottom' in expression.lower()) and len(expression) > 50:
            errorMessage = 'top/bottom expression should be used alone'
            logger.error(errorMessage)
            raise Exception(errorMessage)

        tokens = re.split(r' and | or |[*]', expression, re.IGNORECASE | re.VERBOSE);
        statements = []
        for token in tokens:
            token = token.strip().replace('[', '').replace(']', '')
            if len(token) > 0:
                statements.append(token)
        return statements

    @staticmethod
    def populateIndicators(value, indicators, dataframe):
        np.seterr(all='warn')
        name = value[0] + ' ' + value[1]
        if name not in indicators.keys():
            if value[1] in ['open', 'high', 'low', 'close', 'volume', 'range']:
                if value[1] == 'range':
                    x = "dataframe['" + value[0] + "']['high'] - dataframe['" + value[0] + "']['low']"
                else:
                    x = "dataframe['" + value[0] + "']['" + value[1] + "']"
                #with warnings.catch_warnings():
                #    warnings.filterwarnings('error')
                try:
                    indicators[name] = eval(x)
                except:
                    indicators[name] = None
            else:
                i = value[1].find('(')
                key = value[1][:i]
                parameters = value[1][i+1:]
                if key in TA_MAPPING.keys():
                    mapped = TA_MAPPING[key]
                    x = mapped[0] + '('
                    for j in range(1, len(mapped)-1):
                        x += "dataframe['" + value[0] + "']['" + mapped[j] + "'],"
                    if len(parameters) > 1:
                        if 'macd' in key:   #swap 1st and 2nd arguments for MACD to conform to the usual order of parameters
                            parameters = parameters[:-1].split(',')
                            if key == 'macd':
                                x += parameters[1] + ',' + parameters[0] + ')'
                            else:
                                x += parameters[1] + ',' + parameters[0] + ',' + parameters[2] + ')'
                        elif key == 'median bollinger band':   #the function needs just 1 parameter
                            parameters = parameters[:-1].split(',')
                            x += parameters[0] + ')'
                        else:
                            x += parameters
                    else:
                        x = x[:-1] + ')'
                    #with warnings.catch_warnings():
                    #    warnings.filterwarnings('error')
                    try:
                        indicators[name] = eval(x)  #eval_nb(x)
                    except:
                        indicators[name] = None
                else:
                    logger.warning(f'{value[1]} is undefined in ta_mapping')
                    return
        
    @staticmethod
    def getResults(symbol, timeframes, translation):
        #start = timer()
        if len(translation) == 1 and list(translation.values())[0][0] in [7, 8] and type(list(translation.values())[0][2]) is str:   #IBDRS
            ibdRelativeStrength = 0
            query = f"SELECT ibdRelativeStrength FROM symbols WHERE ticker = '{symbol}'"
            with contextlib.closing(utils.engine.raw_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                row = cursor.fetchone()
                cursor.close()
            if row is not None and row[0] is not None:
                ibdRelativeStrength = row[0]
            return {symbol: ibdRelativeStrength}

        dataframe = {}
        for timeframe, maxPeriod in timeframes.items():
            tablename = timeframe + '_quotes'
            datapoints = maxPeriod * 2 + 50  #500
            #query = "SELECT formatted_date, open, high, low, close, adjclose, volume FROM '{0}' WHERE symbol = '{1}' ORDER BY formatted_date DESC LIMIT 500".format(tablename, symbol[0])
            query = f"SELECT formatted_date as date, open*adjclose/close as open, high*adjclose/close as high, low*adjclose/close as low, adjclose as close, volume FROM {tablename} WHERE symbol = '{symbol}' ORDER BY formatted_date DESC LIMIT {datapoints}"
            query = f"SELECT * FROM ({query}) as quotes ORDER BY date ASC"
            with contextlib.closing(utils.engine.raw_connection()) as conn:
                df = pd.read_sql_query(query, conn, index_col='date')
            #df.info(verbose=True)
            if (df.empty or df.size < 3) and timeframe == 'daily':
                return None
            dataframe[timeframe] = df.round(4)
        #end = timer()
        #logger.debug(f'retrieved data in {str(100*(end-start))} ms')
        #logger.debug('got dataframe')

        if len(translation) == 1 and list(translation.values())[0][0] in [7, 8]:
            indicators = {}
            v = list(translation.values())[0]
            MyScreener.populateIndicators(v[2], indicators, dataframe)
            name = v[2][0] + ' ' + v[2][1]
            if indicators[name] is None or len(indicators[name]) <= v[2][2]:
                return {symbol: 0}
            else:
                return {symbol: getIndicatorValue(indicators, v[2], 1)}

        #start = timer()
        indicators = {}
        for k, v in translation.items():
            if v[0] != 99:
                MyScreener.populateIndicators(v[1], indicators, dataframe)
                if len(v) > 3:
                    if v[0] == 2 and type(v[2]) is list:
                        MyScreener.populateIndicators(v[2], indicators, dataframe)
                    if type(v[3]) is list:
                        MyScreener.populateIndicators(v[3], indicators, dataframe)
        #logger.debug(indicators)
        #end = timer()
        #logger.debug(f'populated indicators in {str(100*(end-start))} ms')
        
        cp_mapping = dict((k.lower(), v) for k,v in utils.get_cp_mapping().items())
        #logger.debug(cp_mapping)
        results = {}
        for k, v in translation.items():
            results[k] = None
            try:
                if v[0] == 1:   #'is above/below'
                    if v[4] is None:
                        v[4] = 1
                    for i in range(1, v[4]+1):
                        name = v[1][0] + ' ' + v[1][1]
                        if indicators[name] is None or len(indicators[name]) <= i:
                            results[k] = False
                            break
                        value1 = getIndicatorValue(indicators, v[1], i)
                        if type(v[3]) is list:
                            index = getIndex(i, v[1][0], v[3][0])
                            name = v[3][0] + ' ' + v[3][1]
                            if indicators[name] is None or len(indicators[name]) <= index:
                                results[k] = False
                                break
                            value2 = getIndicatorValue(indicators, v[3], index)
                        else:
                            value2 = float(v[3])
                        if v[2][0] is None:
                            if v[2][1] == 'above':
                               result = (value1 >= value2)
                            else:
                               result = (value1 <= value2)
                        else:
                            mo = re.search(r'\d+\.?\d*', v[2][0])
                            extra = float(mo.group())
                            if 'more' in v[2][0]:
                                if v[2][1] == 'above':
                                    if '%' in v[2][0]:
                                        result = (value1 >= (value2 + abs(value2) * extra/100))
                                    else:
                                        result = (value1 >= (value2 + extra))
                                else:
                                    if '%' in v[2][0]:
                                        result = (value1 <= (value2 - abs(value2) * extra/100))
                                    else:
                                        result = (value1 <= (value2 - extra))
                            else:
                                if v[2][1] == 'above':
                                    if '%' in v[2][0]:
                                        result = (value1 >= value2 and value1 < (value2 + abs(value2) * extra/100))
                                    else:
                                        result = (value1 >= value2 and value1 < (value2 + extra))
                                else:
                                    if '%' in v[2][0]:
                                        result = (value1 <= value2 and value1 > (value2 - abs(value2) * extra/100))
                                    else:
                                        result = (value1 <= value2 and value1 > (value2 - extra))
                        
                        if not result:
                            results[k] = False
                            break
                        
                    if results[k] is None:
                        results[k] = True
                    
                if v[0] == 2:   #'is in between'
                    if v[4] is None:
                        v[4] = 1
                    for i in range(1, v[4]+1):
                        name = v[1][0] + ' ' + v[1][1]
                        if indicators[name] is None or len(indicators[name]) <= i:
                            results[k] = False
                            break
                        value = getIndicatorValue(indicators, v[1], i)
                        if type(v[2]) is list:
                            index = getIndex(i, v[1][0], v[2][0])
                            name = v[2][0] + ' ' + v[2][1]
                            if indicators[name] is None or len(indicators[name]) <= index:
                                results[k] = False
                                break
                            value1 = getIndicatorValue(indicators, v[2], index)
                        else:
                            value1 = float(v[2])
                        if type(v[3]) is list:
                            index = getIndex(i, v[1][0], v[3][0])
                            name = v[3][0] + ' ' + v[3][1]
                            if indicators[name] is None or len(indicators[name]) <= index:
                                results[k] = False
                                break
                            value2 = getIndicatorValue(indicators, v[3], index)
                        else:
                            value2 = float(v[3])
                        if value1 > value2:  
                            result = (value >= value2 and value <= value1)
                        else:
                            result = (value >= value1 and value <= value2)

                        if not result:
                            results[k] = False
                            break
                        
                    if results[k] is None:
                        results[k] = True
                    
                if v[0] == 3:   #'crossed above/below'
                    if v[4] is None:
                        v[4] = 1
                    for i in range(1, v[4]+1):
                        name = v[1][0] + ' ' + v[1][1]
                        if indicators[name] is None or len(indicators[name]) <= i:
                            results[k] = False
                            break
                        value1 = getIndicatorValue(indicators, v[1], i)
                        value1_1 = getIndicatorValue(indicators, v[1], i+1)
                        if type(v[3]) is list:
                            index = getIndex(i, v[1][0], v[3][0])
                            name = v[3][0] + ' ' + v[3][1]
                            if indicators[name] is None or len(indicators[name]) <= index:
                                results[k] = False
                                break
                            value2 = getIndicatorValue(indicators, v[3], index)
                            value2_1 = getIndicatorValue(indicators, v[3], index+1)
                        else:
                            value2 = float(v[3])
                            value2_1 = value2
                        if v[2] == 'above':
                            result = (value1 >= value2 and value1_1 <= value2_1)
                        else:
                            result = (value1 <= value2 and value1_1 >= value2_1)
                            
                        if result:
                            results[k] = True
                            break
                        
                    if results[k] is None:
                        results[k] = False
                    
                if v[0] in [4, 4.1]:   #['gained', 'dropped']
                    name = v[1][0] + ' ' + v[1][1]
                    if indicators[name] is None or len(indicators[name]) < 3:
                        results[k] = False
                    else:
                        if v[3] is None:
                            v[3] = 1
                        value1 = getIndicatorValue(indicators, v[1], 1)
                        value2 = getIndicatorValue(indicators, v[1], 1+v[3])
                        mo = re.search(r'\d+\.?\d*', v[2])
                        extra = float(mo.group())
                        if 'more' in v[2]:
                            if v[0] == 4:
                                if '%' in v[2]:
                                    result = (value1 >= (value2 + abs(value2) * extra/100))
                                else:
                                    result = (value1 >= (value2 + extra))
                            else:
                                if '%' in v[2]:
                                    result = (value1 <= (value2 - abs(value2) * extra/100))
                                else:
                                    result = (value1 <= (value2 - extra))
                        else:
                            if v[0] == 4:
                                if '%' in v[2]:
                                    result = (value1 >= value2 and value1 < (value2 + abs(value2) * extra/100))
                                else:
                                    result = (value1 >= value2 and value1 < (value2 + extra))
                            else:
                                if '%' in v[2]:
                                    result = (value1 <= value2 and value1 > (value2 - abs(value2) * extra/100))
                                else:
                                    result = (value1 <= value2 and value1 > (value2 - extra))

                        results[k] = (False if result is None else result)
                        
                if v[0] in [5, 5.1]:   #['increasing', 'decreasing']
                    name = v[1][0] + ' ' + v[1][1]
                    for i in range(1, v[2]+1):
                        if indicators[name] is None or len(indicators[name]) <= i:
                            results[k] = False
                            break
                        value1 = getIndicatorValue(indicators, v[1], i)
                        value2 = getIndicatorValue(indicators, v[1], i+1)
                        if v[0] == 5:
                           result = (value1 >= value2)
                        else:
                           result = (value1 <= value2)
                            
                        if not result:
                            results[k] = False
                            break
                            
                    if results[k] is None:
                        results[k] = True
                    
                if v[0] == 6:   #'reached high/low'
                    if v[3] is None:
                        v[3] = 1
                    period = getOffset(v[2].strip().lower(), v[1][0])
                    name = v[1][0] + ' ' + v[1][1]
                    for i in range(1, v[3]+1):
                        if indicators[name] is None or len(indicators[name]) <= i+period:
                            results[k] = False
                            break
                        value1 = getIndicatorValue(indicators, v[1], i)
                        value2 = [getIndicatorValue(indicators, v[1], i+j) for j in range(period)]
                        if 'high' in v[2]:
                            result = (value1 >= max(value2))
                        else:
                            result = (value1 <= min(value2))
                        if result:
                            results[k] = True
                            break
                        
                    if results[k] is None:
                        results[k] = False
                    
                if v[0] == 99:   #formed Candlestick Pattern
                    name = v[2]
                    if 'candlestick pattern' in name:
                        isPatternFound = False
                        sortedByRankMap = sorted(cp_mapping.items(), key=lambda x: x[1][2])
                        if name == 'bullish candlestick pattern':
                            cs_patterns = (kc.lower() for kc,vc in sortedByRankMap if len(vc[0]) > 0 and vc[1] > 0)
                        elif name == 'bearish candlestick pattern':
                            cs_patterns = (kc.lower() for kc,vc in sortedByRankMap if len(vc[0]) > 0 and vc[1] < 0)
                        else:
                            cs_patterns = (kc.lower() for kc,vc in sortedByRankMap if len(vc[0]) > 0 and vc[1] == 0)
                        for cs_pattern in cs_patterns:  #sorted by performance rank
                            isPatternFound = isCandlestickPatternFound(cs_pattern, v[3], dataframe[v[1]], cp_mapping)
                            if isPatternFound:
                                break
                        results[k] = isPatternFound
                    else:
                        results[k] = isCandlestickPatternFound(name, v[3], dataframe[v[1]], cp_mapping)
                    
                #logger.debug(k + ' = ' + str(results[k]))
            except Exception as e:
                logger.error(f'{symbol[0]}: {traceback.format_exc()}')
                return None
                    
        return results        


    @staticmethod
    def sceener(symbol, expression, timeframes, translation):
        utils.engine.dispose()
        result = False
        results = MyScreener.getResults(symbol, timeframes, translation)
        logger.debug(results)
        if len(translation) == 1 and list(translation.values())[0][0] in [7, 8]:
            return results
        if results is not None:
            #logger.debug('expression: ' + expression)
            newexpression = expression
            for k, v in results.items():
                newexpression = newexpression.replace(k, str(v))  
            newexpression = newexpression.replace('[', '(').replace(']', ')').replace('\r', ' ').replace('\n', ' ')
            #logger.debug('newexpression: ' + newexpression)
            try:
                result = eval(newexpression)
            except Exception as e:
                logger.error(f'{newexpression}: {traceback.format_exc()}')
                return None
        logger.debug(f'{symbol}: {str(result)}')
        return symbol if result else None


    @staticmethod
    def calculateIndicator(timeframe, function, dataframe):
        np.seterr(all='warn')
        if function.lower() in ['open', 'high', 'low', 'close', 'volume']:
            return (function, dataframe[timeframe][function])

        i = function.find('(')
        name = function[:i]
        parameters = function[i+1:]
        result = None
        for k, v in TA_MAPPING.items():
            if name.lower() == k:
                x = v[0] + '('
                for j in range(1, len(v)-1):
                    x += "dataframe['" + timeframe + "']['" + v[j] + "'],"
                if len(parameters) > 1:
                    key = name.lower()
                    if 'macd' in key:   #swap 1st and 2nd arguments for MACD to conform to the usual order of parameters
                        parameters = parameters[:-1].split(',')
                        if key == 'macd':
                            x += parameters[1] + ',' + parameters[0] + ')'
                        else:
                            x += parameters[1] + ',' + parameters[0] + ',' + parameters[2] + ')'
                    else:
                        x += parameters
                else:
                    x = x[:-1] + ')'
                #print(x)
                try:
                    result = eval(x)
                except Exception as e:
                    logger.error(f'{x}: {traceback.format_exc()}')
                    result = dataframe[timeframe]['close']
                    result.values[:] = 0
                return (k + function[i:], result)
        
        errorMessage = f'{name} is undefined in ta_mapping'
        logger.error(errorMessage)
        raise Exception(errorMessage)
        

    def checkExpression(self, expression):
        logger.debug('expression: ' + expression)
        #start = timer()
        statements = self.__separate(expression)
        if len(statements) > 0:
            translation = {}
            for statement in statements:
                logger.debug('statement: ' + statement)
                translation[statement] = self.__translate(statement)
                logger.debug('translated: ' + str(translation[statement]))
            #end = timer()
            #logger.debug(f'translation done in {str(100*(end-start))} ms')
            timeframes = self.getTimeframes(translation)

            symbol = 'SPY'      #['FSZ-DBA.TO', 204]  #this is a test case for exception
            #start = timer()
            result = MyScreener.sceener(symbol, expression, timeframes, translation)
            #end = timer()
            #logger.debug(f'got result in {str(100*(end-start))} ms')
            #if len(translation) == 1 and list(translation.values())[0][0] in [7, 8]:
            #    logger.debug(result)
            #else:
            #    result = result is not None
            #    logger.debug(f'result: {str(result)}')
            return translation


    def getTimeframes(self, translationMap):
        timeframes = {}
        for translation in translationMap.values():
            if translation[0] == 99:
                duration = translation[-1]
                if duration is None: 
                    duration = 1
                if translation[1] not in timeframes:
                    timeframes[translation[1]] = duration
                elif duration > timeframes[translation[1]]:
                    timeframes[translation[1]] = duration
            elif translation[0] in [7, 8]:
                if translation[2][0] not in timeframes:
                    timeframes[translation[2][0]] = translation[2][-1]
                elif translation[2][-1] > timeframes[translation[2][0]]:
                    timeframes[translation[2][0]] = translation[2][-1]
            else:
                if translation[1][0] not in timeframes:
                    timeframes[translation[1][0]] = translation[1][-1]
                elif translation[1][-1] > timeframes[translation[1][0]]:
                    timeframes[translation[1][0]] = translation[1][-1]
                if len(translation) > 3:
                    if translation[0] == 2 and type(translation[2]) is list:
                        if translation[2][0] not in timeframes:
                            timeframes[translation[2][0]] = translation[2][-1]
                        elif translation[2][-1] > timeframes[translation[2][0]]:
                            timeframes[translation[2][0]] = translation[2][-1]
                    if type(translation[3]) is list:
                        if translation[3][0] not in timeframes:
                            timeframes[translation[3][0]] = translation[3][-1]
                        elif translation[3][-1] > timeframes[translation[3][0]]:
                            timeframes[translation[3][0]] = translation[3][-1]
        return timeframes


    def __getAllSymbols(self):
        price = None
        if self._priceType is not None and (self._priceLow is not None or self._priceHigh is not None):
            if self._priceType == 0:
                price = 'lastDayPrice'
            elif self._priceType == 1:
                price = 'avg30DayPrice'
            elif self._priceType == 2:
                price = 'avg60DayPrice'
            elif self._priceType == 3:
                price = 'avg90DayPrice'

        volume = None
        if self._volumeType is not None and (self._volumeLow is not None or self._volumeHigh is not None):
            if self._volumeType == 0:
                volume = 'lastDayVolume'
            elif self._volumeType == 1:
                volume = 'avg30DayVolume'
            elif self._volumeType == 2:
                volume = 'avg60DayVolume'
            elif self._volumeType == 3:
                volume = 'avg90DayVolume'

        lastDate = (datetime.today() - timedelta(days=4)).strftime(utils.date_format)   #take into account weekend and holidays
        query = f"SELECT ticker FROM symbols WHERE active=1 and lastDate >= '{lastDate}'"  #only consider symbols that are active and have price up to date
        if self._symbols is not None:
            query += " and ticker in (" + ', '.join(["'%s'" %symbol for symbol in self._symbols]) + ")"
        if price is not None:
            if self._priceLow is not None:
                query += " and " + price + ">=" + str(self._priceLow)
            if self._priceHigh is not None:
                query += " and " + price + "<=" + str(self._priceHigh)
        if volume is not None:
            if self._volumeLow is not None:
                query += " and " + volume + ">=" + str(self._volumeLow)
            if self._volumeHigh is not None:
                query += " and " + volume + "<=" + str(self._volumeHigh)
        if self._industries is not None:
            query += " and industry in (" + ', '.join(["'%s'" %industry for industry in self._industries.split()]) + ")"

        #logger.info(query)
        with contextlib.closing(utils.engine.raw_connection()) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
        return [row[0] for row in rows]


    def getMatchingSymbols(self):
        logger.info('screener_id = ' + str(self._id))
        if self._translation is None or len(self._translation) == 0:
            if self._expression is None or len(self._expression) == 0:
                logger.info('missing expression')
                return
            logger.info('new expression to translate: ' + self._expression)
            statements = self.__separate(self._expression)
            if len(statements) == 0:
                logger.info('no valid statement')
                return
            translation = {}
            for statement in statements:
                translation[statement] = self.__translate(statement)
            logger.debug('translation: ' + str(translation))
            if len(translation) == 0:
                logger.info('no valid translation')
                return
            replaceTranslation(self._id, translation)
            self._translation = translation

        matchingSymbols = []
        symbols = self.__getAllSymbols()
        logger.info(f'#symbols: {str(len(symbols))}')
        if len(symbols) == 0:
            return matchingSymbols

        isTop = False
        isBottom = False
        translationValue = None
        if len(self._translation) == 1:
            translationValue = list(self._translation.values())[0]
            if translationValue[0] == 7:
                isTop = True
            if translationValue[0] == 8:
                isBottom = True

        if (isTop or isBottom) and type(translationValue[2]) is str:   #IBDRS
            query = "SELECT ticker FROM symbols WHERE active=1 and ticker in ({}) order by ibdRelativeStrength {} limit {}" \
                .format(','.join([f"'{symbol}'" for symbol in symbols]), 'desc' if isTop else 'asc', translationValue[1])
            with contextlib.closing(utils.engine.raw_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
                cursor.close()
            matchingSymbols = [row[0] for row in rows]
            logger.info(f'#matchingSymbols: {str(len(matchingSymbols))}')
            return matchingSymbols

        timeframes = self.getTimeframes(self._translation)
        """
        #do in single process
        for symbol in symbols:
            result = MyScreener.sceener(symbol, self._expression, timeframes, self._translation)
            if result is not None:
                matchingSymbols.append(result)
        """     
        #do with multiprocessing
        if __name__ == '__main__':
            parameters = [(symbol, self._expression, timeframes, self._translation) for symbol in symbols]
            processes = mp.cpu_count()   #this process is mainly cpu bound   
            with mp.Pool(processes=processes) as pool:
                results = pool.starmap(MyScreener.sceener, parameters)
            if isTop:
                results = dict((key,d[key]) for d in results for key in d)
                matchingSymbols = heapq.nlargest(int(translationValue[1]), results, key=results.get) 
            elif isBottom:
                results = dict((key,d[key]) for d in results for key in d)
                matchingSymbols = heapq.nsmallest(int(translationValue[1]), results, key=results.get) 
            else:
                for result in results:
                    if result is not None:
                        matchingSymbols.append(result)
                
        logger.info(f'#matchingSymbols: {str(len(matchingSymbols))}')
        return matchingSymbols


def isBlank(myString):
    if myString and myString.strip():
        #myString is not None AND myString is not empty or blank
        return False
    #myString is None OR myString is empty or blank
    return True
  
def getOffset(text, timeframe):
    if len(text) == 0:
        return 0
    mo = re.search(r'\d+', text)
    offset = int(mo.group())
    if timeframe == 'daily':
        if 'week' in text:
            offset *= 5
        elif 'month' in text:
            offset *= 20
    if timeframe == 'weekly':
        if 'day' in text:
            offset //= 5
        elif 'month' in text:
            offset *= 4
    if timeframe == 'monthly':
        if 'day' in text:
            offset //= 20
        elif 'week' in text:
            offset //= 4
    return offset   

def getMaxPeriod(indicator):
    maxPeriod = 1
    i = indicator.find('(')
    if i > 0:
        parameters = indicator[i+1:]
        if len(parameters) > 1:
            parameters = parameters[:-1].split(',')
            for parameter in parameters:
                period = int(float(parameter))
                if period > maxPeriod:
                    maxPeriod = period
    return maxPeriod

def getIndicatorComponents(indicatorString):
    if not ('min' in indicatorString or 'max' in indicatorString or 'avg' in indicatorString):
        functionType = 0
        matches = PLAIN_INDICATOR_RE.search(indicatorString)
        timeframe = matches.group('timeframe')
        if timeframe is None: 
            timeframe = 'daily'
        else:
            timeframe = timeframe.lower()
        indicator = matches.group('indicator')
        #remove extra whitespace in indicator
        name_parameters = indicator.split('(')
        if len(name_parameters) < 2:
            indicator = ' '.join(indicator.split())   
        else:  
            indicator = ' '.join(name_parameters[0].strip().split()) + '(' + ''.join(name_parameters[1].strip().split())  
        offset = matches.group('offset')
        if offset is None: 
            offset = 0
        else:
            offset = getOffset(offset.strip(), timeframe)
        maxPeriod = getMaxPeriod(indicator) + offset
        return [timeframe, indicator, offset, functionType, 0, maxPeriod]
    else:
        if 'min' in indicatorString: 
            functionType = 1
        elif 'max' in indicatorString: 
            functionType = 2
        else: 
            functionType = 3
        matches = AGGREGATE_INDICATOR_RE.search(indicatorString)
        timeframe = matches.group('timeframe')
        if timeframe is None: 
            timeframe = 'daily'
        else:
            timeframe = timeframe.strip()
        indicator = matches.group('indicator')
        #remove extra whitespace in indicator
        name_parameters = indicator.split('(')
        if len(name_parameters) < 2:
            indicator = ' '.join(indicator.split())   
        else:  
            indicator = ' '.join(name_parameters[0].strip().split()) + '(' + ''.join(name_parameters[1].strip().split())  
        functionRange = int(matches.group('range'))
        offset = matches.group('offset')
        if offset is None: 
            offset = 0
        else:
            offset = getOffset(offset.strip(), timeframe)
        maxPeriod = getMaxPeriod(indicator) + functionRange + offset
        return [timeframe, indicator, offset, functionType, functionRange, maxPeriod]

def getIndex(baseIndex, baseTimeframe, timeframe):
    index = baseIndex
    if timeframe != baseTimeframe:   
        #timeframe is different from the base, cast longer to shorter one, the other way doesn't make much sense and is ignored
        if baseTimeframe == 'daily':
            if timeframe == 'weekly':
                index = (baseIndex-1) // 5
            elif timeframe == 'monthly':
                index = (baseIndex-1) // 20
        if baseTimeframe == 'weekly':
            if timeframe == 'monthly':
                index = (baseIndex-1) // 4
    return index

def getIndicatorValue(indicators, indicatorComponents, index):
    name = indicatorComponents[0] + ' ' + indicatorComponents[1]
    if indicatorComponents[3] < 1:  #plain indicator
        value = indicators[name].iloc[-index - indicatorComponents[2]]
    else:
        if indicatorComponents[3] == 1:  #min function
            value = indicators[name].iloc[-index-indicatorComponents[4]-indicatorComponents[2] : -index-indicatorComponents[2]].agg('min')
        elif indicatorComponents[3] == 2:  #max function
            value = indicators[name].iloc[-index-indicatorComponents[4]-indicatorComponents[2] : -index-indicatorComponents[2]].agg('max')
        elif indicatorComponents[3] == 3:  #mean function
            value = indicators[name].iloc[-index-indicatorComponents[4]-indicatorComponents[2] : -index-indicatorComponents[2]].agg('mean')
        else:
            raise Exception(f'Unknown function type {indicatorComponents[3]}')
    return value

def replaceTranslation(screener_id, translationMap):
    with contextlib.closing(utils.engine.raw_connection()) as conn:
        cursor = conn.cursor()
        query = f"DELETE FROM screenertranslation WHERE screener_id = {screener_id}"
        cursor.execute(query)
        newrows = []
        for statement, translation in translationMap.items():
            row = (screener_id, statement, json.dumps(translation))
            newrows.append(row)
        if len(newrows) > 0:
            query = "INSERT INTO screenertranslation (screener_id, statement, translation) VALUES (%s, %s, %s)"
            cursor.executemany(query, newrows)
        conn.commit()
        cursor.close()

def isCandlestickPatternFound(name, duration, df, cp_mapping):
    result = None
    if cp_mapping[name] is None:
        result = False
    else:
        if duration is None:
            duration = 1
        value = getattr(talib, cp_mapping[name][0])(df['open'], df['high'], df['low'], df['close'])
        #logger.debug(name + ': ' + str(value))
        sign_of_value = cp_mapping[name][1]
        for i in range(1, duration+1):
            if sign_of_value > 0 and value.iloc[-i] > 0:
                result = True
                break
            elif sign_of_value < 0 and value.iloc[-i] < 0:
                result = True
                break
            elif sign_of_value == 0 and value.iloc[-i] != 0:
                result = True
                break
            
    if result is None:
        result = False
    return result


def runScreeners(region=None, intraday=False):
    #if intraday:
    #    logging.config.fileConfig("logging_app.cfg")
    #    logger = applogging.getLogger(os.path.basename(__file__))
    logger.info('runScreeners - start')
    myScreeners = []
    with contextlib.closing(utils.engine.raw_connection()) as conn:
        cursor = conn.cursor()
        query = "SELECT id, expression, priceType, priceLow, priceHigh, volumeType, volumeLow, volumeHigh, exchanges, watchlists, industries, lastUpdate FROM screener" 
        if region is not None:
            query += " WHERE region = '{}'".format(region) 
        query += " ORDER BY id" 
        cursor.execute(query)
        screeners = cursor.fetchall()
        for screener in screeners:
            if intraday:   #run newly created or updated screeners only
                if screener[0] >= 6:  #always include defaultScreeners to copy results from when user created screeners from sample records
                    query = f"SELECT lastUpdate FROM screenerresult WHERE screener_id = {screener[0]}" 
                    cursor.execute(query)
                    result = cursor.fetchone()
                    if result is not None and result[0] > screener[-1]:
                        #print(f'screener {screener[0]} skipped')
                        continue

            #print(f'screener {screener[0]} is going to be run')
            myScreener = MyScreener()
            watchlists = screener[9]
            if isBlank(watchlists):    #watchlists take precedence to exchanges
                myScreener.exchanges = screener[8]
            else:
                query = f"SELECT symbols FROM watchlist where id in ({watchlists.replace(' ',',')})"
                cursor.execute(query)
                rows = cursor.fetchall()
                symbols = set()
                for row in rows:
                    symbols.update(row[0].split(' '))
                myScreener.symbols = symbols
            if not isBlank(myScreener.exchanges) or myScreener.symbols is not None:
                myScreener.id = screener[0]
                myScreener.expression = screener[1]
                myScreener.priceType = screener[2]
                myScreener.priceLow = screener[3]
                myScreener.priceHigh = screener[4]
                myScreener.volumeType = screener[5]
                myScreener.volumeLow = screener[6]
                myScreener.volumeHigh = screener[7]
                myScreener.industries = screener[10]

                query = f"SELECT statement, translation FROM screenertranslation where screener_id = {myScreener.id}"
                cursor.execute(query)
                rows = cursor.fetchall()
                translation = {}
                for row in rows:
                    translation[row[0]] = json.loads(row[1])
                myScreener.translation = translation
                myScreeners.append(myScreener)
        cursor.close()

    defaultScreeners = []
    for myScreener in myScreeners:
        message = None
        query_result = None
        if myScreener.id < 6:
            defaultScreeners.append(myScreener)
            if intraday: 
                continue  #don't run defaultScreeners intraday
            #query_result = f"SELECT result FROM screenerresult WHERE screener_id = {myScreener.id}" 
        else:
            for ds in defaultScreeners:
                if (myScreener.expression == ds.expression and myScreener.exchanges == ds.exchanges and myScreener.industries == ds.industries and 
                    myScreener.priceType == ds.priceType and myScreener.priceLow == ds.priceLow and myScreener.priceHigh == ds.priceHigh and 
                    myScreener.volumeType == ds.volumeType and myScreener.volumeLow == ds.volumeLow and myScreener.volumeHigh == ds.volumeHigh):  
                    query_result = f"SELECT result FROM screenerresult WHERE screener_id = {ds.id}" 
                    break;
        if query_result is not None:  #copy result from defaultScreeners when criteria totally match
            with contextlib.closing(utils.engine.raw_connection()) as conn:
                cursor = conn.cursor()
                cursor.execute(query_result)
                result = cursor.fetchone()
                cursor.close()
                if result is not None:
                    message = result[0]
                 
        if message is None: 
            if myScreener.symbols is None and not isBlank(myScreener.exchanges):
                query = f"SELECT ticker FROM symbols WHERE active=1 and exchange_id in ({myScreener.exchanges.replace(' ',',')})" 
                with contextlib.closing(utils.engine.raw_connection()) as conn:
                    cursor = conn.cursor()
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    cursor.close()
                myScreener.symbols = [row[0] for row in rows]
            if len(myScreener.symbols) == 0:
                continue
            matchingSymbols = myScreener.getMatchingSymbols()
            utils.engine.dispose()
            if len(matchingSymbols) > 0:
                message = 'Matching symbols: ' + ' '.join(matchingSymbols)
            else:
                message = 'No matching symbols'
                    
        #logger.info(f'screener_id = {myScreener.id}, message = {message}')
        with contextlib.closing(utils.engine.raw_connection()) as conn:
            cursor = conn.cursor()
            query = f"SELECT user_id, name FROM screener WHERE id = {myScreener.id}"
            cursor.execute(query)
            screener = cursor.fetchone()
            screener_name = ''
            email = None
            if screener is not None:
                user_id = screener[0]
                screener_name = screener[1]
                if user_id != 1:  #send email to non system user
                    query = f"SELECT email FROM user WHERE id = {user_id} and isVerified = 1"
                    cursor.execute(query)
                    result = cursor.fetchone()
                    if result is not None:
                        email = result[0]
                
                #query = "UPDATE screener SET result = %s, resultTimestamp = %s WHERE id = %s" 
                query = "INSERT INTO screenerresult (screener_id, result) VALUES (%s, %s) ON DUPLICATE KEY UPDATE result=%s, lastUpdate=UTC_TIMESTAMP()"
                cursor.execute(query, (myScreener.id, message, message))
                conn.commit()
                
            cursor.close()
                
        if email is not None:  
            subject = f"Result of screener [{screener_name}]"
            message += utils.mail_signature
            utils.sendMail(email, subject, message, logger)
    logger.info('runScreeners - end')


def testScreener(id, symbols=None):
    myScreener = MyScreener()
    with contextlib.closing(utils.engine.raw_connection()) as conn:
        cursor = conn.cursor()
        query = f"SELECT id, expression, priceType, priceLow, priceHigh, volumeType, volumeLow, volumeHigh, exchanges, watchlists, industries FROM screener WHERE id = {id}" 
        cursor.execute(query)
        screener = cursor.fetchone()
        if symbols is not None:
            myScreener.symbols = symbols
        else:
            watchlists = screener[9]
            if isBlank(watchlists):    #watchlists take precedence to exchanges
                myScreener.exchanges = screener[8]
            else:
                query = f"SELECT symbols FROM watchlist where id in ({watchlists.replace(' ',',')})"
                cursor.execute(query)
                rows = cursor.fetchall()
                symbols = set()
                for row in rows:
                    symbols.update(row[0].split(' '))
                myScreener.symbols = symbols

        if myScreener.symbols is not None or not isBlank(myScreener.exchanges):
            myScreener.id = screener[0]
            myScreener.expression = screener[1]
            myScreener.priceType = screener[2]
            myScreener.priceLow = screener[3]
            myScreener.priceHigh = screener[4]
            myScreener.volumeType = screener[5]
            myScreener.volumeLow = screener[6]
            myScreener.volumeHigh = screener[7]
            myScreener.industries = screener[10]

            query = f"SELECT statement, translation FROM screenertranslation where screener_id = {myScreener.id}"
            cursor.execute(query)
            rows = cursor.fetchall()
            translation = {}
            for row in rows:
                translation[row[0]] = json.loads(row[1])
            myScreener.translation = translation

        if myScreener.symbols is None and not isBlank(myScreener.exchanges):
            query = f"SELECT ticker FROM symbols WHERE active=1 and exchange_id in ({myScreener.exchanges.replace(' ',',')})" 
            cursor.execute(query)
            rows = cursor.fetchall()
            myScreener.symbols = [row[0] for row in rows]
        cursor.close()

    if len(myScreener.symbols) > 0:
        matchingSymbols = myScreener.getMatchingSymbols()
    if len(matchingSymbols) > 0:
        print(id, symbols, 'Matching symbols: ' + ' '.join(matchingSymbols))
    else:
        print(id, symbols, 'No matching symbols')
     

def main():
    region = None
    intraday = False
    if len(sys.argv) >= 2:
        try:
            opts, args = getopt.getopt(sys.argv[1:], "r:i")
        except getopt.GetoptError:
            print(f'Usage: {os.path.basename(__file__)} [-r|-i] [<region>|<intraday>]')
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-r", "--region"):
                region = arg
            elif opt in ("-i", "--intraday"):
                intraday = True
            
        if region is not None:
            region = utils.regions.get(int(region))
            if region is None:
                region = 'Americas'

    runScreeners(region, intraday)
    
    
if __name__ == '__main__':  
    main()
    """
    filter1 = 'Close 3 days ago has been more than 15% above weekly MA(50) 1 month ago for the last 2 weeks'
    filter1_1 = 'Volume MA(90) is above 100000.0'
    filter1_2 = 'Close is from 1.0 to 1999.9'
    filter2 = '[EMA(10) 2 days ago crossed above 50 or EMA(10) 2 days ago crossed above EMA(50) within the last 5 days]'
    filter3 = 'weekly EMA(10) 1 week ago dropped more than 30% over the last 1 month'
    filter4 = '[close 1 day ago is below EMA(23) or close 1 day ago is below EMA(30)]'
    filter5 = 'EMA(10) 1 week ago reached a new 10 weeks high within the last 6 days'
    expression = filter1 + '\nand ' + filter1_1 + '\nand ' + filter1_2+ '\nand ' + filter2 + '\nand ' + filter3 + '\nand ' + filter4 + '\nand ' + filter5
    
    with contextlib.closing(utils.engine.raw_connection()) as conn:
        cursor = conn.cursor()
        query = "SELECT expression FROM screener WHERE id = 1" 
        cursor.execute(query)
        row = cursor.fetchone()
        cursor.close()
    expression = row[0]
     
    #expression = 'Median Bollinger Band (20.0,  2.5) has been increasing for 20 days'     #'bottom   20 IBD  Relative   Strength'
    #expression = 'Bullish candlestick pattern formed within the last 3 days and close is below EMA(10) and close is above 5 and RSI(7) is below 45 and MA(50) is above MA(200) and MA(50) is above MA(50) 50 days ago'
    #expression = 'avg(volume, 22) is from volume ma(10) 1 week ago to 9999999999 and MACD(12,   26,9) has crossed above MAX ( MACD  Signal ( 12, 26, 9 ), 10) within the last 5 days and Close is more than 5% above MA(10) for the last 10 days and  Bullish Harami   Cross formed    within the last  2 days'   
    #expression = 'Volume MA ( 20 ) is above 100000 and MA(60) is above 20 and Range is above MIN ( Range, 6 ) 1 day ago and High is below High 1 day ago and Low is above Low 1 day ago and MIN(CCI(10),5) is below -100 and Aroon Up(63) is above Aroon Down(63)'
    myScreener = MyScreener()
    myScreener.checkExpression(expression)
    """   
    #testScreener(1, ['AEM'])


