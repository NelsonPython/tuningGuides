# Experiments:  Automate Publishing Code Snippets

## 1. Experiment 1

Using the  guide, create a template to structure content so that each group of code snippets and their corresponding instructions can be maintained separately.  Develop a Python script to assemble these markdown files into one markdown file for loading into the DevZone.  For purposes of these experiments, this is called the "DevZone markdown"

### File contents:

000-config.json - contains the order in which the code and text snippets should be arranged

001_title.md - the title may not be included in the DevZone markdown file.  Only ## level two headers and lower may be used.

002_intro.md - contains all content prior to the first code snippet

#### Instructions and code snippet files:
There are multiple blocks of code and two files for each block:  
- step-by-step instructions along with code
- block of code that a developer can cut-and-paste directly into the command line

Test data for this experiment is two actual code blocks from our most complex tuning guide:
```
003_Configuring_the_Slurm_Workload_Manager.md
004_summary.md

005_Updating_node_resource_information.md
006_summary.md
```
