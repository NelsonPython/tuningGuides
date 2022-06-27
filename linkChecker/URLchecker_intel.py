import os
import urllib.request
from datetime import datetime

class IntelURLchecker():
    '''
        Get the authors of Intel tuning guides
        Print the Intel error log
    '''

    def __init__(self):
        pass

    def getAuthors(self, authorFile):
        '''
        load a dictionary with author's email addresses and the page they are responsible for
        use this to lookup the email address when a broken link must be reported
        '''
        import csv
        with open(authorFile, mode='r') as infile:
            reader = csv.reader(infile)
            authors = {rows[0]:rows[1] for rows in reader}
        return authors


    def openLog(self,logName):
        return open(logName,"w")

    def printIt(self, line, msg, filename, logFile, authors):
        # lookup author
        author_email = authors[filename]
        print("AUTHOR: ", author_email, file=logFile)
        print('HTMLFILE: ',filename, file=logFile)    
        print("SENTENCE: ",line, file=logFile)
        print("MESSAGE: ", msg, "\n", file=logFile)

        #print("AUTHOR: ", author_email)
        #print('HTMLFILE: ',filename)    
        #print("SENTENCE: ",line)
        #print("MESSAGE: ", msg, "\n")

    def closeLog(self, logFile, K, b):
        print("Total links followed: ", K, file=logFile)
        print("Total broken links: ", b, file=logFile)
        print("======================================================", file=logFile)
        logFile.close()


if __name__ == "__main__":
    print("HERE")
