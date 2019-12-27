import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

subscription_key = "70e4e05c78354be3b0cb45ca5efb3460"

search_url = "https://api.cognitive.microsoft.com/bing/v7.0/videos"
search_term = "puppies"
