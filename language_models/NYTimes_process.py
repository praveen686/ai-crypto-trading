# Demo code sample. Not indended for production use.

# See instructions for installing Requests module for Python
# https://requests.readthedocs.io/en/master/user/install/#install

import requests
import json
import os

filenames = os.listdir('NYTimes_origin/NYTimes/')

def toJson(finename):
    with open('NYTimes_origin/NYTimes/' + filename) as fin:
      res_json = json.load(fin)
    info = res_json['response']['docs']
    l = []
    s = []
    types = ['World', 'U.S.', 'Business Day', 'Technology']
    for line in info:
    	if (line['section_name'] in types):
            s.append(line['section_name'])
            l.append({'section_name' : line['section_name'], 'pub_date': line['pub_date'], 'headline': line['headline']['main']})
    s = set(s)
    print(filename, len(l), s)
    with open('NYTimes_processed/'+filename, 'w') as fout:
    	json.dump(l, fout)

#for filename in filenames:
#    toJson(filename)

filenames= os.listdir('NYTimes_processed/')
res_json = []
for filename in filenames:
    with open('NYTimes_processed/' + filename) as fin:
      res_json += list(json.load(fin))
print(len(res_json))
with open("NYTimes_all.json", 'w') as fout:
    json.dump(res_json, fout)
l = []
import string
import re
punctuation_string = string.punctuation
for line in res_json:
    s = line['headline']
    s = re.sub('[{}]'.format(punctuation_string),"",s)
    s = s.replace('\u2019', '')
    s = s.replace('\u2018', '')
    l.append(s)
with open("NYTimes_all_headline.txt", 'w') as fout:
    for line in l:
        fout.write(line + "\n")

#print(res_json[0]['section_name'])
#print(res_json[0]['pub_date'])
#print(res_json[0]['headline'])
