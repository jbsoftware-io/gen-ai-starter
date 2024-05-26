import json


def get_city_data():
    """Get city data from json file."""

    with open("./etc/cities.json") as f:
        cities = json.load(f)
    return sorted(set([f"{city["city"]}, {city["state"]}" for city in cities]))
