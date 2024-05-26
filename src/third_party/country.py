import json


def get_country_data():
    """Get country data from json file."""

    with open("./etc/countries.json") as f:
        all_country_data = json.load(f)
        countries_by_name = {
            country["name"]["common"]: country for country in all_country_data
        }
        # Add blank option
        countries_by_name['Select'] = {
            "flags": {
                "png": countries_by_name['United States']['flags']['png'],
            },
            "capital": ["None"],
            "region": "None",
            "subregion": "None",
            "population": "None",
            "area": "None",
        }

        return countries_by_name
