import requests

# Make a GET request to the API
response = requests.get('https://api3.geo.admin.ch/rest/services/api/MapServer/ch.bfe.solarenergie-eignung-daecher/legend') #'https://api3.geo.admin.ch/rest/services/api/MapServer/ch.bfe.solarenergie-eignung-daecher') #'https://api3.geo.admin.ch/rest/services/api/MapServer')

# Convert the response to JSON
type(response)
data = response.json()
data

head = requests.head('https://api3.geo.admin.ch/rest/services/api/MapServer/ch.bfe.solarenergie-eignung-daecher')
head
type(head)

options = requests.options('https://api3.geo.admin.ch/rest/services/api/MapServer/ch.bfe.solarenergie-eignung-daecher')
options


data['id']
data['name']
data['fields']