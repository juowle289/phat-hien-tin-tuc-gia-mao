import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sentence_transformers import SentenceTransformer

# Embeddings model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

class FakeNewsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fake News Detection GUI")
        self.root.geometry("950x750")
        self.root.configure(bg="#eef2f3")

        self.fake_file = None
        self.true_file = None
        self.vectorizer_choice = tk.StringVar(value="tfidf")
        self.best_model = None
        self.vectorizer = None

        # Tiêu đề chính
        tk.Label(root, text="📰 PHÁT HIỆN TIN TỨC GIẢ MẠO",
                 font=("Arial", 22, "bold"), bg="#2c3e50", fg="white",
                 pady=15).pack(fill="x")

        # --- Chọn file ---
        file_frame = tk.LabelFrame(root, text="Chọn dữ liệu huấn luyện", 
                                   font=("Arial", 12, "bold"),
                                   bg="#eef2f3", padx=10, pady=10)
        file_frame.pack(pady=10, fill="x")

        tk.Button(file_frame, text="📂 Chọn file Fake.csv", command=self.load_fake,
                  bg="#e67e22", fg="white", width=20).grid(row=0, column=0, padx=5, pady=5)
        self.fake_label = tk.Label(file_frame, text="Chưa chọn file Fake.csv", bg="#eef2f3")
        self.fake_label.grid(row=0, column=1, sticky="w")

        tk.Button(file_frame, text="📂 Chọn file True.csv", command=self.load_true,
                  bg="#27ae60", fg="white", width=20).grid(row=1, column=0, padx=5, pady=5)
        self.true_label = tk.Label(file_frame, text="Chưa chọn file True.csv", bg="#eef2f3")
        self.true_label.grid(row=1, column=1, sticky="w")

        # --- Chọn vectorizer ---
        vec_frame = tk.LabelFrame(root, text="Phương pháp vector hóa", 
                                  font=("Arial", 12, "bold"),
                                  bg="#eef2f3", padx=10, pady=10)
        vec_frame.pack(pady=10, fill="x")

        tk.Radiobutton(vec_frame, text="Bag of Words", variable=self.vectorizer_choice, 
                       value="bow", bg="#eef2f3", font=("Arial", 11)).pack(anchor="w")
        tk.Radiobutton(vec_frame, text="TF-IDF", variable=self.vectorizer_choice, 
                       value="tfidf", bg="#eef2f3", font=("Arial", 11)).pack(anchor="w")
        tk.Radiobutton(vec_frame, text="Sentence Embeddings", variable=self.vectorizer_choice, 
                       value="emb", bg="#eef2f3", font=("Arial", 11)).pack(anchor="w")

        # Train button
        tk.Button(root, text="🚀 Train & Evaluate", command=self.train_and_evaluate,
                  bg="#2980b9", fg="white", font=("Arial", 13, "bold"),
                  width=20, height=2).pack(pady=10)

        # --- Kết quả đánh giá ---
        tk.Label(root, text="📊 Kết quả đánh giá mô hình", 
                 font=("Arial", 14, "bold"), bg="#eef2f3", fg="black").pack()
        self.result_text = scrolledtext.ScrolledText(root, height=15, width=110, font=("Consolas", 10))
        self.result_text.pack(pady=10)

        # --- Dự đoán tin tức mới ---
        pred_frame = tk.LabelFrame(root, text="Nhập tin tức cần kiểm tra", 
                                   font=("Arial", 12, "bold"),
                                   bg="#eef2f3", padx=10, pady=10)
        pred_frame.pack(pady=10, fill="both", expand=True)

        self.input_text = scrolledtext.ScrolledText(pred_frame, height=6, width=90, font=("Arial", 11))
        self.input_text.pack(pady=5)

        tk.Button(pred_frame, text="🔍 Dự đoán", command=self.predict_text, 
                  bg="#8e44ad", fg="white", font=("Arial", 12, "bold"), width=15).pack(pady=5)

        self.pred_label = tk.Label(pred_frame, text="", font=("Arial", 13, "bold"), bg="#eef2f3")
        self.pred_label.pack()

    def load_fake(self):
        self.fake_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.fake_file:
            self.fake_label.config(text=f"Đã chọn: {self.fake_file}")

    def load_true(self):
        self.true_file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.true_file:
            self.true_label.config(text=f"Đã chọn: {self.true_file}")

    def safe_show(self):
        """Hiển thị biểu đồ mà không chặn Tkinter"""
        plt.show(block=False)
        plt.pause(1)

    def train_and_evaluate(self):
        if not self.fake_file or not self.true_file:
            messagebox.showerror("Lỗi", "Vui lòng chọn đủ cả 2 file Fake.csv và True.csv")
            return

        # ===== 1. ĐỌC DỮ LIỆU =====
        fake_df = pd.read_csv(self.fake_file, encoding="ISO-8859-1", on_bad_lines="skip")
        true_df = pd.read_csv(self.true_file, encoding="ISO-8859-1", on_bad_lines="skip")

        fake_df["label"] = 0
        true_df["label"] = 1
        data = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42).reset_index(drop=True)

        # Xác định cột văn bản
        if "text" in data.columns:
            X_raw = data["text"].astype(str)
        else:
            cols = [c for c in data.columns if data[c].dtype == object]
            X_raw = data[cols].astype(str).agg(" ".join, axis=1)

        y = data["label"]

        # Hàm tiền xử lý
        def clean_text(text):
            import re
            text = text.lower()
            text = re.sub(r"http\S+|www\S+", " ", text)
            text = re.sub(r"[^a-z\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        X = X_raw.apply(clean_text)
        data["_len"] = X.apply(lambda x: len(x.split()))

        # ===== 2. KHÁM PHÁ DỮ LIỆU =====
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, ">>> KHÁM PHÁ DỮ LIỆU\n")
        self.result_text.insert(tk.END, f"- Tổng số mẫu: {len(data)}\n")
        self.result_text.insert(tk.END, f"- Số tin giả (Fake=0): {(data['label']==0).sum()}\n")
        self.result_text.insert(tk.END, f"- Số tin thật (True=1): {(data['label']==1).sum()}\n")
        self.result_text.insert(tk.END, f"- Độ dài trung bình văn bản: {round(data['_len'].mean(), 2)} từ\n\n")

        # Biểu đồ phân phối nhãn
        sns.countplot(x="label", hue="label", data=data, palette="Set2", legend=False)
        plt.title("Phân phối nhãn (0=Fake, 1=True)")
        self.safe_show()

        # Biểu đồ phân phối độ dài văn bản
        plt.figure(figsize=(7,5))
        sns.histplot(data["_len"], bins=50, kde=True, color="purple")
        plt.title("Phân phối độ dài văn bản (số từ)")
        plt.xlabel("Số từ")
        plt.ylabel("Số mẫu")
        self.safe_show()

        # ===== 3. VECTOR HÓA =====
        method = self.vectorizer_choice.get()
        if method == "bow":
            self.vectorizer = CountVectorizer(stop_words="english", max_features=30000, min_df=10, max_df=0.8)
            X_vec = self.vectorizer.fit_transform(X)
        elif method == "tfidf":
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=30000, min_df=10, max_df=0.8)
            X_vec = self.vectorizer.fit_transform(X)
        else:
            X_vec = embedder.encode(X.tolist(), show_progress_bar=True)

        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

        # In số lượng mẫu train/test (tổng + chi tiết từng nhãn)
        self.result_text.insert(tk.END, ">>> CHIA DỮ LIỆU\n")
        self.result_text.insert(tk.END, f"- Số mẫu train: {X_train.shape[0]} ({X_train.shape[0]/len(data)*100:.1f}%)\n")
        self.result_text.insert(tk.END, f"   + Fake (0): {(y_train==0).sum()}\n")
        self.result_text.insert(tk.END, f"   + True (1): {(y_train==1).sum()}\n")
        self.result_text.insert(tk.END, f"- Số mẫu test : {X_test.shape[0]} ({X_test.shape[0]/len(data)*100:.1f}%)\n")
        self.result_text.insert(tk.END, f"   + Fake (0): {(y_test==0).sum()}\n")
        self.result_text.insert(tk.END, f"   + True (1): {(y_test==1).sum()}\n\n")

        # ===== 4. HUẤN LUYỆN MÔ HÌNH =====
        models = {
            "Naive Bayes": MultinomialNB() if method != "emb" else None,
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "SVM": LinearSVC(max_iter=3000),
            "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=50, random_state=42)
        }

        results = {}
        for name, model in models.items():
            if model is None:
                continue
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Fake", "True"], yticklabels=["Fake", "True"])
            plt.title(f"Confusion Matrix - {name}")
            plt.xlabel("Dự đoán")
            plt.ylabel("Thực tế")
            self.safe_show()

            self.result_text.insert(tk.END, f"{name}:\n Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}\n\n")

        # ===== 5. SO SÁNH =====
        df_results = pd.DataFrame(results).T
        df_results.plot(kind="bar", figsize=(8,6))
        plt.title("So sánh các mô hình")
        plt.ylabel("Score")
        plt.xticks(rotation=0)
        self.safe_show()

        # Lưu model tốt nhất
        best_model_name = df_results["Accuracy"].idxmax()
        self.result_text.insert(tk.END, f"👉 Mô hình tốt nhất: {best_model_name} (Accuracy={df_results['Accuracy'].max():.4f})\n")
        self.best_model = models[best_model_name]

    def predict_text(self):
        if self.best_model is None:
            messagebox.showwarning("Cảnh báo", "Bạn cần train mô hình trước khi dự đoán")
            return

        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập văn bản để dự đoán")
            return

        method = self.vectorizer_choice.get()
        if method == "emb":
            vec = embedder.encode([text])
        else:
            vec = self.vectorizer.transform([text])

        pred = self.best_model.predict(vec)[0]
        result = "✅ Tin thật" if pred == 1 else "❌ Tin giả"
        self.pred_label.config(text=f"Kết quả: {result}", fg="green" if pred == 1 else "red")


if __name__ == "__main__":
    root = tk.Tk()
    app = FakeNewsApp(root)
    root.mainloop()
