import os
import urllib.request
import URLchecker_intel

class URLcrawl():
    '''
        Check for broken links
            - read the stream and ignore everything before:  <div class="editorialBody">
                 and ignore everything after:  <"/main">
             - follow each URL and record the response code
    '''

    def __init__(self):
        pass

    def saveHTMLfile(self, url, saveHTMLfile, text):
        ''' TODO - this has never been tested'''
        if saveHTMLfile == 1:
            fileURL = "0-intel_tuning_guides/"+url
            print(httpURL," COPIED TO: ",fileURL,"\n")
            fo = open(fileURL, "w", encoding='utf8')
            print(text, file=fo)
            fo.close()

    def getHTML(self, httpURL):
        user_agent = 'Mozilla/5.0'
        headers = {'User-Agent':user_agent,} 
        request = urllib.request.Request(httpURL,None,headers)
        response = urllib.request.urlopen(request)
        data = response.read()
        text = data.decode("utf8")
        response.close()
        # parsed[0] will be the name of the file such as: lammps-tuning-guide.html
        parseURL = httpURL.split("/")
        parsed = parseURL[-1].split("#")
        lines = text.split('\n')
        return parsed, lines

    def getLinks(self,httpURL,readFlag):
        '''
            receive a link to the starting HTML page
            skip the header and footer and only read the content
            parse URLs that have <a href
            DO NOT get links to images
            return a list of <a href...URLs found
        '''

        parsed, lines = self.getHTML(httpURL)
        urls = []
        for line in lines:
            url = 'none'
            if '<div class="editorialBody">' in line:
                # start looking for URLs
                readFlag = 1

            if '</main>' in line:
                # stop looking for URLs
                readFlag = 0

            if readFlag == 1:
                # evaluate Intel specific content
                if '<a href' in line:
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
                        url = "https://www.intel.com/content/www/us/en/developer/articles/guide/" + httpURL + url

                if url != 'none':
                    urls.append(url)
        return urls

    def parseHTML(self, httpURL, saveHTMLfile, readFlag, verbose, logFile, authors, intelURLchecker, b='', msg='', totK=0, broken=0):
        '''
            receive the link to each page in the list or URLs
            get HTML
            parse URLs out of the HTML
            go to every link (including images) in the content of the page between the header and footer
        '''
        print(httpURL)
        parsed, lines = self.getHTML(httpURL)        
        for line in lines:
            if '<div class="editorialBody">' in line:
                # start looking for URLs
                readFlag = 1

            if '</main>' in line:
                # stop looking for URLs
                readFlag = 0

            if readFlag == 1:
                if '<a href' in line:
                    totK+=1
                    a = line.split('<a href="')
                    a = a[1].split('">')
                    url = a[0]

                    if url.startswith('/content/www/') or url.startswith('/content/dam/'):
                        url = "https://www.intel.com" + url

                    if url.startswith('/file'):
                        url = "https://www.intel.com" + url

                    if '"' in url:
                        # looking for title=" or target="
                        u = url.split('"')
                        url = u[0]

                    if url[0] == '#':
                        url = "https://www.intel.com/content/www/us/en/developer/articles/guide/" + httpURL + url

                    b, msg = self.followURL(url,verbose)
                    broken += int(b)
                    if msg != '999':
                        intelURLchecker.printIt(line, msg, parsed[0], logFile, authors)

                if 'src="' in line:
                    # check the link to the images
                    p = line.split('src="')
                    p = p[1].split(' width')
                    url = "https://www.intel.com" + p[0][:-1]
                    if "etc.clientlibs" not in url:
                        totK+=1
                        b, msg = self.followURL(url,verbose)
                        broken += int(b)
                    if msg != '999':
                        intelURLchecker.printIt(line, msg, parsed[0], logFile, authors)        

        return broken, totK

    def followURL(self,url,verbose, msg='',reformat='',broken=0):
        '''
            receive a possible URL
            clean up the URL
            follow the URL and get the HTTP status
            save broken links to the log file
        '''
        
        if '"/>' in url:
            reformat = "BAD URL: "+url+","
            u = url.split('"')
            url = u[0]
            reformat += " reformatATTED: "+url+","

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
            msg = reformat+" MALFORMED URL: "+url+", "+" ERROR: ",str(e)+","
            broken = 1
            return broken, msg

        try:
            response = urllib.request.urlopen(request)
        except urllib.error.HTTPError as e:
            # print specific HTTP errors
            if str(e.code) == '403':
                if verbose == 1:
                    msg = reformat+str(e.code)+" "+url+","
                    return broken, msg
                else:
                    # return 999 because verbose != 1 and valid links should not print
                    return broken, '999'
            else:                
                msg = reformat+str(e.code)+" "+url+","
                broken = 1
                return broken, msg
        except urllib.error.URLError as e:
            # print other errors
            msg = reformat+str(e.reason)+" "+url+","
            broken = 1
            return broken, msg
        else:
            # if no errors then HTTP 200 is expected
            if verbose == 1:
                res = response.getcode()
                response.close()
                msg = reformat+str(res)+" "+url+","
                return broken, msg
            else:
                # return 999 because verbose != 1 and valid links should not print
                return broken, '999'

if __name__ == "__main__":
    print("HERE")
