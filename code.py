import urllib2

proxy_support = urllib2.ProxyHandler({"http":"http://uat2-vip.web.gs.com:83"})
opener = urllib2.build_opener(proxy_support)
urllib2.install_opener(opener)

html = urllib2.urlopen("https://www.zebapi.com/api/v1/market/ticker/btc/inr").read()
print html

import requests
r = requests.post('https://www.unocoin.com/oauth/token', 
              data={'grant_type':'client_credentials','access_lifetime':'7200'}, 
              auth=('5O9ZQNO0NU7', '6ca97a44-553b-45b1-a052-39f0df8640f3'))
print(r.text)
