These are the preliminary scripts used to test the idea of automated link checking.  Stored temporarily for reference.

Version 1.0 has a hard-coded list of URLs from the Tuning Guide landing page

VERSION 1.0
python3 intel_robot_link_checker.py
output goes to tuningGuideLink.log

TODO - rewrite the comparison because using sets scrambles the result set and makes it difficult to read

``` python3 compareLog.py ```
