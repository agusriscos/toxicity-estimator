import os
from bs4 import BeautifulSoup
import requests, json

if __name__ == '__main__':
    print("Ejecutando desde:", os.getcwd())
    resp = requests.get("http://www.netlingo.com/acronyms.php")
    soup = BeautifulSoup(resp.text, "html.parser")
    slang_dict = {}
    key = ""
    value = ""

    for div in soup.findAll('div', attrs={'class':'list_box3'}):
        for li in div.findAll('li'):
            for a in li.findAll('a'):
                key = a.text
                value = li.text.split(key)[1]
                slang_dict[key.lower()] = value.lower()

    with open('../data/utils/internet_slang.json', 'w') as f:
        json.dump(slang_dict, f, indent=2)