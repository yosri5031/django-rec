from transformers import pipeline

classifier = pipeline("zero-shot-classification")

options = {
    "emotions": ["Love and romance", "Happiness and joy", "Peace and tranquility", "Inspiration and motivation", "Sentimental and nostalgic"],
    "occasion": ["Birthday", "Anniversary", "Housewarming", "Holiday celebration", "Graduation", "Retirement", "Baby shower", "Engagement", "Prom", "Valentine's Day", "New Year's Eve", "Easter", "Mother's Day", "Father's Day", "Wedding", "Bachelorette party", "Bachelor party", "Job promotion", "Thank you", "Get well soon", "Sympathy", "Congratulations", "Good luck", "Just because"],
    "interests": ["Architecture", "Cars & Vehicules", "Religious", "Fiction", "Tools", "Human Organes", "Symbols", "Astronomy", "Plants", "Animals", "Art", "Celebrities", "Flags", "HALLOWEEN", "Quotes", "Sports", "Thanksgiving", "Maps", "Romance", "Kitchen", "Musical Instruments", "Black Lives Matter", "Cannabis", "Vegan", "Birds", "Dinosaurs", "rock and roll", "Firearms", "Dances", "Sailing", "Jazz", "Christmas", "Greek Methology", "Life Style", "Planes", "Vintage", "Alphabets", "Weapons", "Insects", "Games", "JEWELRY", "Science", "Travel", "Cats", "Circus", "Lucky charms", "Wild West", "Dogs"],
    "audience": ["Child Audience", "Teen Audience", "Adult Audience", "Senior Audience"],
    "personality": ["Casual and laid-back", "Elegant and sophisticated", "Edgy and avant-garde", "Bohemian and free-spirited", "Classic and timeless"]
}

category_order = ["emotions", "occasion", "interests", "audience", "personality"]

def classify_text(text):
    result = classifier(text, options["emotions"] + options["occasion"] + options["interests"] + options["audience"] + options["personality"])
    predicted_labels = []
    seen_labels = set()
    for label in result["labels"]:
        category = None
        for key, values in options.items():
            if label in values:
                category = key
                break
        if category and category not in seen_labels:
            predicted_labels.append({"category": category, "value": label})
            seen_labels.add(category)
        if len(predicted_labels) == 5:
            break
    return predicted_labels