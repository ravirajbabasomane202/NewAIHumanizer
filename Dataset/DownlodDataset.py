from datasets import load_dataset

# 1. Load the dataset (ensure you are logged in if it's private)
ds = load_dataset("silentone0725/ai-human-text-detection-v1")

# 2. Save the dataset to a local folder
ds.save_to_disk("./my_local_dataset")

# 3. To load it back later from that folder:
# from datasets import load_from_disk
# ds_local = load_from_disk("./my_local_dataset")
