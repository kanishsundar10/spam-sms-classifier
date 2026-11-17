# spam-sms-classifier
NLP project that classifies messages as spam or not spam using TF-IDF and Naive Bayes, with an interactive prediction mode (Python, scikit-learn)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

messages = [
    ("Congratulations! You won a free iPhone. Click now", "spam"),
    ("Win cash prize now, claim your reward today", "spam"),
    ("Free entry in lottery, reply YES to claim", "spam"),
    ("Urgent! Your account is blocked, verify now", "spam"),
    ("Get 50% discount, limited time offer, buy now", "spam"),
    ("You have been selected for a free gift voucher", "spam"),
    ("Earn money from home, no investment needed", "spam"),
    ("Claim your free recharge now", "spam"),
    ("Hi, are we meeting at college tomorrow?", "ham"),
    ("Please send me the assignment notes", "ham"),
    ("Lunch ku varegala? I am waiting at canteen", "ham"),
    ("Happy birthday! Have a great day", "ham"),
    ("Class is postponed to 10 AM tomorrow", "ham"),
    ("Can you call me when you are free?", "ham"),
    ("Submit the record book before Friday", "ham"),
    ("Thanks for helping me with the project", "ham"),
]

texts = [m[0] for m in messages]
labels = [m[1] for m in messages]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("clf", MultinomialNB()),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

print("\n=== Spam Checker (type 'exit' to quit) ===")
while True:
    text = input("Enter message: ")
    if text.lower() == "exit":
        break
    print("Prediction:", model.predict([text])[0].upper())
