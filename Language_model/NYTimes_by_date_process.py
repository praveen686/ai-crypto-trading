import json
import string
import re
punctuation_string = string.punctuation
with open("NYTimes_all.json", 'r') as fin:
    dataset = json.load(fin)

def get_info(line):
    time = line['pub_date']
    headline = line['headline']
    year = time[:4]
    month = time[5:7]
    day = time[8:10]
    return year+month+day, headline
i = 1
time, headline = get_info(dataset[0])
dataset_by_day = []
while (i < len(dataset)):
    line = dataset[i]
    new_time, new_headline = get_info(line)
    #print(new_time, new_headline)
    if new_time == time:
        headline += (' ' + new_headline)
    else:
        headline = re.sub('[{}]'.format(punctuation_string),"",headline)
        headline = headline.replace('\u2019', '')
        headline = headline.replace('\u2018', '')
        dataset_by_day.append({"pub_date" : time, "headline": headline})
        s = headline.split()
        if len(s) >= 512:
            print(time)
        headline = new_headline
        time = new_time
    i += 1
with open("NYTimes_all_by_halfdate.json", 'w') as fout:
    json.dump(dataset_by_day, fout)
#{"section_name": "Business Day",
#"headline": "House Prices Rise Again, but the Pace Could Slow",
#"pub_date": "2014-01-01T00:01:56+0000"}
