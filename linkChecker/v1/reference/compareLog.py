##########################
# References:
# https://stackoverflow.com/questions/28213525/python-compare-2-files-and-output-differences
##########################

import datetime

def compare(benchmark,daily,newErrorLog,dt,msg):
    with open(benchmark,'r') as f:
        d=set(f.readlines())

    with open(daily,'r') as f:
        e=set(f.readlines())

    fo = open(newErrorLog,'w')
    write_dt_msg = dt + " " + msg + "\n\n"
    fo.write(write_dt_msg)
    # the benchmark file should be empty if all links were published correctly
    # the daily file will have more errors than the benchmark
    # as links break over time
    for line in list(d-e):
        fo.write(line)
        fo.write("\n\n")
    fo.write("==========\n\n")
    fo.close()

########################
benchmark = '/home/nelson/TuningGuides/0-resources/benchmark_errors.log'
daily = '/home/nelson/TuningGuides/0-log/tuningGuideLink.log'
newErrorLog = '/home/nelson/TuningGuides/0-log/new_errors.log'
dt = datetime.date.today()
msg = "---------BENCHMARK - DAILY = FEWER BROKEN LINKS"
compare(benchmark,daily,newErrorLog,str(dt),msg)
msg = "+++++++++DAILY - BENCHMARK = MORE BROKEN LINKS"
compare(daily,benchmark, newErrorLog,str(dt),msg)
