import requests

API_KEY = "a83e882f08331cb9c7552aa15813b8a2"  
city = input("Enter city name: ")  

url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
response = requests.get(url)
weather_data = response.json()

if response.status_code == 200:
    print(f"{city} Weather: {weather_data['main']['temp']}°C")
else:
    print(" Error fetching weather data:", weather_data)
