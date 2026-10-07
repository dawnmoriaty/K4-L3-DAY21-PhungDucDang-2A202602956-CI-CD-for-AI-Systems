import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # -------------------------------------------------------------------------
    # BONUS 5: Canh bao lech lac du lieu (Data Drift Alert)
    # Tinh ty le lop duong trong tap train. Neu lech > 5% so voi 24.8%, in canh bao.
    # -------------------------------------------------------------------------
    pos_ratio = float(y_train.mean())
    ref_ratio = 0.248
    if abs(pos_ratio - ref_ratio) > 0.05:
        print(f"WARNING (Data Drift): Ty le lop duong = {pos_ratio:.4f} lech > 5% so voi tham chieu {ref_ratio:.4f}!")
    else:
        print(f"Data Drift Check: Ty le lop duong = {pos_ratio:.4f} (an toan quanh tham chieu {ref_ratio:.4f})")

    with mlflow.start_run():
        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # -------------------------------------------------------------------------
        # BONUS 2: Dieu chinh nguong quyet dinh (Decision Threshold Tuning)
        # Quet nguong tu 0.1 den 0.9 (buoc 0.05) de tim nguong dat F1 toi uu.
        # -------------------------------------------------------------------------
        probs = model.predict_proba(X_eval)[:, 1]
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_threshold = 0.5
        best_f1 = f1
        for th in thresholds:
            th_preds = (probs >= th).astype(int)
            th_f1 = float(f1_score(y_eval, th_preds))
            if th_f1 > best_f1:
                best_f1 = th_f1
                best_threshold = float(round(th, 2))

        # -------------------------------------------------------------------------
        # BONUS 3: Bao cao Precision / Recall tu dong & Confusion Matrix
        # -------------------------------------------------------------------------
        cm = confusion_matrix(y_eval, preds)
        clf_report = classification_report(y_eval, preds, target_names=["<=50K (Lop 0)", ">50K (Lop 1)"])
        detail_text = (
            "====================================================================\n"
            "BÁO CÁO CHI TIẾT ĐỘ ĐO PHÂN LOẠI & CONFUSION MATRIX (BONUS 3)\n"
            "====================================================================\n\n"
            f"1. Confusion Matrix (Ma tran nham lan):\n{cm}\n\n"
            f"2. Classification Report (Precision / Recall / F1):\n{clf_report}\n\n"
            "3. Decision Threshold Tuning (Bonus 2):\n"
            f"   - Nguong mac dinh 0.5: F1 = {f1:.4f}\n"
            f"   - Nguong toi uu {best_threshold}: F1 = {best_f1:.4f} (Cai thien: {best_f1 - f1:+.4f})\n\n"
            "4. Data Drift Analysis (Bonus 5):\n"
            f"   - Ty le lop duong tap train: {pos_ratio:.4f} (Moc tham chieu: {ref_ratio:.4f})\n"
        )
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write(detail_text)

        # Log metrics vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("positive_class_ratio", pos_ratio)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.log_artifact("outputs/detail.txt")
        mlflow.sklearn.log_model(model, "model")

        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f} | Best Threshold: {best_threshold} (F1: {best_f1:.4f})")

        # Luu report.json gom day du chi so cho Quality Gate va bao cao
        report_data = {
            "f1_score": f1,
            "accuracy": acc,
            "positive_class_ratio": pos_ratio,
            "best_threshold": best_threshold,
            "best_f1_score": best_f1,
        }
        with open("outputs/report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
