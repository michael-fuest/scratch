from pathlib import Path

class FileManager():

    root = Path(__file__).parents[2]
    test = f"{root}/test"
    source = f"{root}/src"
    test_data = f"{test}/data"
    test_data_input = test_data + "/Housing.csv"

