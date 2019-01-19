import requests 

photo_url = 'https://avatars.io/twitter/nntaleb'
Picture_request = requests.get(photo_url)

if Picture_request.status_code == 200:
    with open("/Users/rvg/Documents/springboard_ds/springboard_portfolio/CNN_eyeglasses/data/twitter/nntaleb.jpg", 'wb') as f:
        f.write(Picture_request.content)