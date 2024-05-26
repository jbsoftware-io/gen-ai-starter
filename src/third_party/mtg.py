import json
import random
from mtgsdk import Card


def get_card_data(card_name: str = ""):
    """scrape information from magic the gathering public api."""

    if not card_name:
        # get a random card name
        pageSize = 100
        pageNum = random.randint(0, 800)
        card_subset = Card.where(page=pageNum).where(pageSize=pageSize).all()
        card = card_subset[random.randint(0, pageSize)]
    else:
        similar_cards = Card.where(name=card_name).all()
        card = similar_cards[random.randint(0, len(similar_cards))]

    # only include subset of fields, LLM context is small
    card = {
        "name": card.name,
        "layout": card.layout,
        "mana_cost": card.mana_cost,
        "cmc": card.cmc,
        "colors": card.colors,
        "color_identity": card.color_identity,
        "type": card.type,
        "original_type": card.original_type,
        "supertypes": card.supertypes,
        "subtypes": card.subtypes,
        "types": card.types,
        "rarity": card.rarity,
        "text": card.text,
        "artist": card.artist,
        "number": card.number,
        "power": card.power,
        "toughness": card.toughness,
        "printings": card.printings,
        "image_url": card.image_url,
        "set": card.set,
        "set_name": card.set_name
    }

    data = json.dumps(card, separators=(',', ':'))

    return (data, card)
