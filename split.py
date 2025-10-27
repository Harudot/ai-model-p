import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/Projects/python/ai/character dataset/HMCC letters merged.csv")

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("train size:", len(X_train))
print("test size:", len(X_test))

train_data = pd.concat([y_train, X_train], axis=1)
test_data = pd.concat([y_test, X_test], axis=1)

train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print("✅ Files saved:")
print(" - train_data.csv")
print(" - test_data.csv")

LABELS = {
   1: "а",
   2: "б",
   3: "в",
   4: "г",
   5: "д",
   6: "е",
   7: "ё",
   8: "ж",
   9: "з",
   10: "и",
   11: "й",
   12: "к",
   13: "л",
   14: "м",
   15: "н",
   16: "о",
   17: "ө",
   18: "п",
   19: "р",
   20: "с",
   21: "т",
   22: "у",
   23: "ү",
   24: "ф",
   25: "х",
   26: "ц",
   27: "ч",
   28: "ш",
   29: "щ",
   30: "ъ",
   31: "ь",
   32: "ы",
   33: "э",
   34: "ю",
   35: "я"
}

example_label = y_train.iloc[0]
print("Example numeric label:", example_label)
print("Example character label:", LABELS.get(example_label, "?"))
