import requests
import sys
import re
import csv
import random
 
    
def writerCsv(res_dict):
    fileHeader = []
    max_len = 0
    
    data = [0] * len(res_dict)
    for key, value in res_dict.items():
        fileHeader.append(key)
        max_len = max(max_len, len(value))
    csvFile = open("confidence.csv", "w", newline='', encoding='utf8')
    writer = csv.writer(csvFile)
    writer.writerow(fileHeader)
    for i in range(max_len):
        for j in range(len(data)):
            if len(res_dict[fileHeader[j]]) > i:
                data[j] = res_dict[fileHeader[j]][i] 
                
            else:
                data[j] = "None"
        writer.writerow(data)
    
    
    csvFile.close()
res_dict ={
    "s":[1,2,3],"j":[2,3]
}

writerCsv(res_dict)