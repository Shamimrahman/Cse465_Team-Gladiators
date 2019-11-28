import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

subscription_key = "your-subscription-key"
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
search_term = "puppies"
