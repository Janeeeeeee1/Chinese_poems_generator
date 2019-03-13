import os
import json
import re

filePath = './json/'
saveFile = open('poem_fanti.txt','w',encoding='utf-8')

for file in os.listdir(filePath):
    if os.path.isfile(os.path.join(filePath,file)) and re.match("(.*)(\.)(json)",file) != None:
        print("processing file %s" %file)
        poems = json.load(os.path.join(filePath,file),'r',encoding='utf-8')
        for poem in poems:
            author = poem['author']
            content = ''.join(poem['paragraphs'])
            saveFile.write(author + ':' + content + '\n')

saveFile.close()
