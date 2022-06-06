from sklearn.model_selection import train_test_split


def triple_split(splits, *args):
    train_X, temp_X, train_y, temp_y, train_stratify, temp_stratify = train_test_split(*args, stratify test_size=0.9, stratify=stratify, shuffle = True)
    cal_X, test_X, cal_y, test_y, cal_stratify, test_stratify = train_test_split(temp_X, temp_y, temp_stratify, test_size=0.7, stratify=temp_stratify, shuffle = True)

