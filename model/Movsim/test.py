import csv


def get_loc(gps_id):
    y = gps_id // 50
    x = gps_id % 50
    return [round(110.07 + y * (110.42 - 110.07) / 50, 3), round(19.31 + x * (20.04 - 19.31) / 50, 3)]


data = []
gps_id = 1
while gps_id < 2500:
    data.append(get_loc(gps_id))
    gps_id = gps_id + 1

with open(r'data/HaiNan/gps', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the data
    writer.writerows(data)

with open(r'data/HaiNan/gps') as input_file:
    s = input_file.read().replace(',', ' ')
    with open(r'data/HaiNan/gps', 'w') as output_file:
        output_file.write(s)

# import pandas as pd
# from sklearn.model_selection import train_test_split

# data = pd.read_csv(r"data/HaiNan/real1.data")
# def train_test_val_split(data, ratio_train, ratio_test, ratio_val):
#     train, middle = train_test_split(data, train_size=ratio_train, test_size=ratio_test + ratio_val)
#     ratio = ratio_val/(1-ratio_train)
#     test, validation = train_test_split(middle, test_size=ratio)
#     return train, test, validation

# train, test, validation = train_test_val_split(data, 0.6, 0.2, 0.2)
# train.to_csv(r'data/HaiNan/train1.data', index=False)
# test.to_csv(r'data/HaiNan/test1.data', index=False)
# validation.to_csv(r'data/HaiNan/val1.data', index=False)
