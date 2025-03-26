class AmazonCat13K:
    name = "AmazonCat-13K"
    file_name = "AmazonCat-13K.zip"
    url = "https://drive.google.com/u/0/uc?id=17rVRDarPwlMpb3l5zof9h34FlwbpTu4l"
    train_file = "trn.json.gz"
    test_file = "tst.json.gz"
    label_file = "Yf.txt"
    label_content_col = None
    target_col = "target_ind"
    title_col = "title"
    content_col = "content"
    encoding = "latin-1"
    num_classes = 13_330
    jsonl = True
    train_lf_filter = None
    test_lf_filter = None

class EURLex4K:
    name = "EURLex-4K"
    file_name = "EURLex4K.zip"
    url = "https://drive.usercontent.google.com/u/0/uc?id=1xR0v57p8yUn-G1Mzs7Bi2Lt4D4KoRHmF"
    train_file = "trn.json.gz"
    test_file = "tst.json.gz"
    label_file = "Yf.txt"
    label_content_col = None
    target_col = "target_ind"
    title_col = None
    content_col = "content"
    encoding = "utf-8"
    num_classes = 3956
    jsonl = True
    train_lf_filter = None
    test_lf_filter = None

class Wiki1031K:
    name = "Wiki10-31K"
    file_name = "Wiki10-31K.zip"
    url = "https://drive.usercontent.google.com/u/0/uc?id=12kpXHV07RgOFarW7ug0nTJcXjfjTNVJS&export=download"
    train_file = "trn.json.gz"
    test_file = "tst.json.gz"
    label_file = "Yf.txt"
    label_content_col = None
    target_col = "target_ind"
    title_col = None
    content_col = "content"
    encoding = "utf-8"
    num_classes = 30938
    jsonl = True
    train_lf_filter = None
    test_lf_filter = None



def load_dataset(name: str):
    datasets = [
        AmazonCat13K,
        EURLex4K,
        Wiki1031K,
    ]

    assert name in [dataset.name for dataset in datasets], f"Unknown dataset: {name}"

    for dataset in datasets:
        if dataset.name == name:
            return dataset()
