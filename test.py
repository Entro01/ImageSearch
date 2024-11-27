import requests

url = 'http://reverseimagesearch.eba-4yjczqvr.us-east-1.elasticbeanstalk.com/find_similar/?top=3'

files = {'image': open(r'C:\Users\shubh\Downloads\test_img.webp', 'rb')}

response = requests.post(url, files=files)

print(response.status_code)
print(response.text)