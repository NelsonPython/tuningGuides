##################
# Check for broken links
# - read the stream and ignore everything before:  <div class="editorialBody">
#     and ignore everything after:  <"/main">
# - follow each URL and record the response code
# - when verbose is 0 report response codes other than 200 or 403
#       when verbose is 1 report all response codes
# - match HTML page name with author
#       report the: author's email, html file name, sentence, and error
#
# TODO:
# 1. group the errors under author email and html page to avoid redundancy
# 2. Parse the tuning guide title from this tag:
# <h1 class="editorialTitle"> Deep Learning with Intel® AVX512 and Intel® Deep Learning Boost Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors </h1>
# 3. Intel redirects
#
# authors = urlcrawl.getAuthors('C:\\Users\\SDene\\AppData\\Local\\Programs\\Python\\Python37\\0-intel_tuning_guides\\authors\\pages2authors.csv')
# logFile = urlcrawl.openLog("C:\\Users\\SDene\\AppData\\Local\\Programs\\Python\\Python37\\0-intel_tuning_guides\\log\\tuningGuideLink.log")
# urls = [[args['URL'], args['guideName']]]
#
# REFERENCES:
# 1. https://stackoverflow.com/questions/24153519/how-to-read-html-from-a-url-in-python-3
#       sample:  url = "http://www.memidex.com/"
# 2. https://stackoverflow.com/questions/1726402/in-python-how-do-i-use-urllib-to-see-if-a-website-is-404-or-200
# 3. https://stackoverflow.com/questions/60977494/how-can-i-find-if-a-word-string-occurs-more-than-once-in-an-input-list-in-pyth
#########################

import os
import urllib.request
import argparse
import URLcrawl
import URLchecker_intel

#ap = argparse.ArgumentParser()
#ap.add_argument("-u", "--URL", required=True, help="Find all the links in this URL and create a list.  Check all the links in all webpages in the list.")
#ap.add_argument("-n", "--guideName", required=True, help="Name of Guide")
#ap.add_argument("-v", "--verbose", required=True, help="0-errors only; 1-all results")
#args = vars(ap.parse_args())
#used for testing:
#args = {'verbose':1, 'URL': 'https://www.intel.com/content/www/us/en/developer/articles/guide/lammps-tuning-guide.html', 'guideName': 'LAMMPS Tuning Guide on 3rd Generation Intel® Xeon® Scalable Processors Based Platform'}
#args = {'verbose':0, 'URL': 'https://www.intel.com/content/www/us/en/developer/articles/guide/xeon-performance-tuning-and-solution-guides.html', 'guideName': 'Intel® Xeon® Performance Tuning and Solutions'}

args = {'verbose':0, 'URL': 'https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html', 'guideName': 'Intel® Distribution of OpenVINO™ Toolkit'}
args = {'verbose':0, 'URL': 'https://www.intel.com/content/www/us/en/homepage.html', 'guideName': 'Intel®'}


verbose = args['verbose']
startPage = args['URL']
logFilename = "/home/nelson/TuningGuides/0-log/aemTest.log"
authorsFilename = '/home/nelson/TuningGuides/0-resources/pages2authors.csv'
readFlag = 0

urlcrawl = URLcrawl.URLcrawl()
intelURLchecker = URLchecker_intel.IntelURLchecker()

authors = intelURLchecker.getAuthors(authorsFilename)
urls = urlcrawl.getLinks(startPage, readFlag)

logFile = intelURLchecker.openLog(logFilename)

HTMLurls = []
otherURLS = []
for url in urls:
    if url.count("http") == 1 and 'html' in url:
        # only HTML pages with an URL that contains only one instance of "http" can be checked
        HTMLurls.append(url)
    else:
        otherURLS.append(url)

saveHTMLfile = 0
# 0-do not save; 1-save the file
K = 0
# counts the total number of links followed
brokenK = 0
# counts the number of broken links found
for url in HTMLurls:
    # parse HTML, follow links, and report HTTP status
    try:
        b,k = urlcrawl.parseHTML(url,saveHTMLfile,readFlag,verbose,logFile, authors, intelURLchecker)
        brokenK += int(b)
        K += int(k)
    except Exception as e:
        print("Exception: ", url, " in ", e)

intelURLchecker.closeLog(logFile, K, brokenK)
