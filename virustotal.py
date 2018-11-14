#!/usr/bin/env python

import hashlib, urllib, urllib2, json, os, requests, time
import pandas as pd

def handleHTTPErros(code):
    errmsg = 'Something went wrong. Please try again later, or contact us.'
    if code == 404:
        print errmsg + '\n[Error 404].'
        return 0
    elif code == 403:
        print 'You do not have permissions to make that call.\nThat should not have happened, please contact us.\n[Error 403].'
        return 0
    elif code == 204:
        print 'The quota limit has exceeded, please wait and try again soon.\nIf this problem continues, please contact us.\n[Error 204].'
        return 0
    else:
        print errmsg + '\n[Error '+str(code)+']'
    return 0

# Sending and scanning URLs
def scan_url(link):
    api_key = '0ae6b8d3275c989958651b00a6bd285d610baa16fe2b7bed2e56b2f5bcdae5bf' #bryan: my specific api key
    url = 'https://www.virustotal.com/vtapi/v2/url/scan'
    parameters = {"url": link, "apikey": api_key}
    #data = urllib.urlencode(parameters)
    #req = urllib2.Request(url, data)
    """
    try:
        response_1 = requests.post(url, data=parameters)
        print str(response_1)
        if str(response_1) == '<Response [200]>':
            response = urllib2.urlopen(req)
            xjson = response.read()
            response_code = json.loads(xjson).get('response_code')
            verbose_msg = json.loads(xjson).get('verbose_msg')
            #scan_id = response.json().get('scan_id')
            if response_code == 1:
                print verbose_msg
            else:
                print verbose_msg
        elif str(response_1) == '<Response [204]>':
            xjson = "bad"
            print "darn"
    """

    response = requests.post(url, data=parameters)
    print response
    if str(response) == '<Response [200]>':
        xjson = response.json()
        print "json: ", xjson
        response_code = xjson.get('response_code')
        verbose_msg = xjson.get('verbose_msg')
        print "response code: ", response_code
    elif str(response) == '<Response [204]>':
        time.sleep(5)
        xjson = "empty"

    '''
    try:
        #response = urllib2.urlopen(req)
        #print "Response: ", response
        #xjson = response.read()
        #response_code = json.loads(xjson).get('response_code')
        #verbose_msg = json.loads(xjson).get('verbose_msg')
        if response_code == 1:
            print verbose_msg
        else:
            print verbose_msg
    except Exception as e:
        xjson = "bad"
        print "bad"

    except urllib2.HTTPError, e:
        handleHTTPErros(e.code)
    except urllib2.URLError, e:
        print 'URLError: ' + str(e.reason)
    except Exception:
        import traceback
        print 'generic exception: ' + traceback.format_exc()
    '''
    return xjson

def scan_report(xjson):
    api_key = '0ae6b8d3275c989958651b00a6bd285d610baa16fe2b7bed2e56b2f5bcdae5bf' #bryan: my specific api key
    #scan_id = json.loads(xjson).get('scan_id')
    scan_id = xjson.get('scan_id')
    url = 'https://www.virustotal.com/vtapi/v2/url/report'
    parameters = {'apikey': api_key, 'resource': str(scan_id)}
    #data = urllib.urlencode(parameters)
    #req = urllib2.Request(url, data)

    response = requests.post(url, params=parameters)
    print response
    if str(response) == '<Response [200]>':    
        json_report = response.json()
    elif str(response) == '<Response [204]>':
        time.sleep(5)
        json_report = "empty"
    else:
        json_report = "empty"
    return json_report

    '''
    try:
        response = urllib2.urlopen(req)
        json_report = response.read()
        num_total_scans = json.loads(json_report).get('total')
        scans = json.loads(json_report).get('scans')
        return json_report
    except urllib2.HTTPError, e:
        handleHTTPErros(e.code)
    except urllib2.URLError, e:
        print 'URLError: ' + str(e.reason)
    except Exception:
        import traceback
        print 'generic exception: ' + traceback.format_exc()
    '''

def detection(json_report):
    '''
    try:
        #num_total_scans = json.loads(json_report).get('total')
        #scans = json.loads(json_report).get('scans')

    except urllib2.HTTPError, e:
        handleHTTPErros(e.code)
    except urllib2.URLError, e:
        print 'URLError: ' + str(e.reason)
    except Exception:
        import traceback
        print 'generic exception: ' + traceback.format_exc()
    '''
    num_total_scans = json_report.get('total')
    scans = json_report.get('scans')
    for key in scans.keys():
        detection = scans[key][u'detected'] #careful with u
        if (detection == True):
            virus_count += 1
            if (virus_count > 2):
                malicious = True
                return malicious
    malicious = False
    return malicious

def parse_csv():
    data = pd.read_csv('pin.csv', header=None, usecols=[4], names=['url'])
    urls_to_scan = data.values.tolist()
    return urls_to_scan
    #data_df = pd.DataFrame(data)



num_req = 0
urls_to_scan = parse_csv()
for url in urls_to_scan:
    if num_req == 4:
        time.sleep(60)
        num_req = 0
    xjson = scan_url(url[0]) # because url is a list with one element in it so use url[0]
    num_req += 1

    json_report = scan_report(xjson)
    num_req += 1
    print num_req
    malicious = detection(json_report)
    if malicious == False:
        print "NOT Malicious"
    else:
        print "Malicious"

    print "------------------------------------------"

    if xjson == "empty" or json_report == "empty":
        print "Empty\n"
        continue

