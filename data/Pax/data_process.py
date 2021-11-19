import pandas as pd

# TODO:
#   1. assign a new ID to each item, and the item ID starts with 0 or 1, and
#       save the new_id -> raw_id mapping
#   2. assign a new ID to each user, and the user ID starts with 0 or 1, and
#       save the new_id -> raw_id mapping
#   3. generate a dictionary that key is the user's new ID, value is a list of interacted items
#       sorted by time
#   4. generate a dictionary, that key is the user's new ID, value is (pretest, posttest) tuple

data = pd.read_csv("rec_opened.csv")
data = data[["rec_index", "timestamp", "user_id"]]
rec_open_data = data.sort_values(by='timestamp', ascending=True)
for i, (rec_index, timestamp, user) in rec_open_data.iterrows():
    print(rec_index, timestamp, user)
