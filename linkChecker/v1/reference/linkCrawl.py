##########################
# References:
# https://stackoverflow.com/questions/24153519/how-to-read-html-from-a-url-in-python-3
# sample:  url = "http://www.memidex.com/"
# https://stackoverflow.com/questions/1726402/in-python-how-do-i-use-urllib-to-see-if-a-website-is-404-or-200
##########################

import os
import urllib.request

def followURL(url,verbose):
    msg = ''
    reform = ''
    broken = 0
    
    if '"/>' in url:
        reform = "BAD URL: "+url+","
        u = url.split('"')
        url = u[0]
        reform += " REFORMATTED: "+url+","

    # removing dots at the end
    if url[-1:len(url)] == '.':
        url = url[:-1]

    # remove clutter after .html or .pdf
    if url[:-4] != 'html' or url[:-3] != 'pdf':
        url_list = url.split(".")
        prefx = url_list[-1].split("%")
        url = ''

        for x in range(len(url_list)-1):
            url += url_list[x]+"."
        url+= prefx[0]
            
    # pick one user_agent and add it to the header.  possible user agents:
    # user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    user_agent = 'Mozilla/5.0'
    headers = {'User-Agent':user_agent,}

    stopFlag = 0
    try:
        request = urllib.request.Request(url,None,headers)
    except Exception as e:
        msg = reform+" MALFORMED URL: "+url+", "+" ERROR: ",str(e)+","
        broken = 1
        return broken, msg

    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as e:
        # print specific HTTP errors
        if str(e.code) == '403':
            if verbose == 1:
                msg = reform+str(e.code)+" "+url+","
                return broken, msg
            else:
                # return 999 because verbose != 1 and valid links should not print
                return broken, '999'
        else:                
            msg = reform+str(e.code)+" "+url+","
            broken = 1
            return broken, msg
    except urllib.error.URLError as e:
        # print other errors
        msg = reform+str(e.reason)+" "+url+","
        broken = 1
        return broken, msg
    else:
        # if no errors then HTTP 200 is expected
        if verbose == 1:
            res = response.getcode()
            response.close()
            msg = reform+str(res)+" "+url+","
            return broken, msg
        else:
            # return 999 because verbose != 1 and valid links should not print
            return broken, '999'


def printIt(line, msg, filename, logFile):
    # lookup author
    author_email = authors[filename]
    print("AUTHOR: ", author_email, file=logFile)
    print('HTMLFILE: ',filename, file=logFile)    
    print("SENTENCE: ",line, file=logFile)
    print("MESSAGE: ", msg, file=logFile)

##################
# find the HTTP status of all URLs in all guides in /0-intel_tuning_guides folder
# open each HTML page and ignore everything before:  <div class="editorialBody">
#     and ignore everything after:  <"/main">
# follow each URL and record the response code
# when verbose is 0 report response codes other than 200 or 403
# when verbose is 1 report all response codes
# match HTML page name with author
# report the: author's email, html file name, sentence, and error
# TODO:
# - find missing author emails
# - group the errors under author email and html page to avoid redundancy
##################

verbose = 0
# counts the total number of links followed
K = 0
# sums the number of broken links
b = 0
# receives the broken link counter update from the function
brokenK = 0
# HTTP status message
msg = ''
# skip the header and footer and only read the content
readFlag = 0

# load a dictionary with author's email addresses and the page they are responsible for
# use this to lookup the email address when a broken link must be reported
import csv
with open('/home/nelson/TuningGuides/0-resources/pages2authors.csv', mode='r') as infile:
    reader = csv.reader(infile)
    authors = {rows[0]:rows[1] for rows in reader}   

# check every URL in the content
# the test folder has one html with known broken links
#guidesFolder = '/home/nelson/TuningGuides/0-test'
guidesFolder = '/home/nelson/TuningGuides/0-intel_tuning_guides'

fileFolders = os.listdir(guidesFolder)
logFile = open("tuningGuideLink.log","w")

for filename in fileFolders:
    if filename.endswith(".html"):
        file2open = guidesFolder + "/" + filename
        with open(file2open, encoding='utf8') as file:

            text = file.read()
            lines = text.split('\n')

            for line in lines:
                if '<div class="editorialBody">' in line:
                    # start looking for URLs
                    readFlag = 1

                if '</main>' in line:
                    # stop looking for URLs
                    readFlag = 0

                if readFlag == 1:
                    if '<a href' in line:
                        K+=1
                        a = line.split('<a href="')
                        a = a[1].split('">')
                        url = a[0]

                        if url.startswith('/content/www/') or url.startswith('/content/dam/'):
                            url = "https://www.intel.com" + url

                        if url.startswith('/file'):
                            url = "https://www.intel.com" + url

                        if '"' in url:
                            # title=", target="
                            u = url.split('"')
                            url = u[0]

                        if url[0] == '#':
                            url = "https://www.intel.com/content/www/us/en/developer/articles/guide/" + filename + url

                        brokenK, msg = followURL(url,verbose)
                        b += int(brokenK)
                        if msg != '999':
                            printIt(line, msg, filename, logFile)

                    if 'src="' in line:
                        p = line.split('src="')
                        p = p[1].split(' width')
                        url = "https://www.intel.com" + p[0][:-1]
                        if "etc.clientlibs" not in url:
                            K+=1
                            brokenK, msg = followURL(url,verbose)
                            b += int(brokenK)
                        if msg != '999':
                            printIt(line, msg, filename, logFile)

print("Total links followed: ", K, file=logFile)
print("Total broken links: ", b, file=logFile)
print("======================================================", file=logFile)
logFile.close()
