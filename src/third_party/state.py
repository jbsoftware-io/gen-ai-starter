import json


def get_state_data():
    with open("./etc/cities.json") as f:
        cities = json.load(f)
    return sorted(set([city["state"] for city in cities]))
