#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch pEC50 Predictor GUI
=========================

A single-file desktop application for batch pEC50 prediction using a fixed
pretrained model and a fixed feature configuration.

Main features
-------------
- Import Excel or CSV files.
- Read all worksheets from Excel files or a single table from CSV files.
- Normalize fingerprint column names automatically.
- Select the final 60 fingerprint features plus ExactMass according to the
  bundled JSON feature configuration.
- Predict pEC50 values in batch for multiple compounds.
- Export one Excel file containing per-sheet prediction results and a summary
  sheet.

Packaging layout
----------------
This script expects the following internal resources:
- assets/svr_model.joblib
- assets/feature_config.json

The resource resolver supports both normal Python execution and PyInstaller
packaging.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# =============================================================================
# Resource paths
# =============================================================================
def resource_base_dir() -> Path:
    """
    Return the base directory for bundled resources.

    Development mode:
        Directory of this script.

    PyInstaller mode:
        Temporary unpacked directory stored in sys._MEIPASS.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


BASE_DIR = resource_base_dir()
MODEL_PATH = BASE_DIR / "assets" / "svr_model.joblib"
CONFIG_PATH = BASE_DIR / "assets" / "feature_config.json"


# =============================================================================
# Core prediction utilities
# =============================================================================
def normalize_fp_name(col: object) -> object:
    """
    Normalize fingerprint column names to the form fp_<integer>.

    Examples
    --------
    3138       -> fp_3138
    "3138"     -> fp_3138
    "fp_3138"  -> fp_3138

    Non-fingerprint names are returned unchanged.
    """
    if pd.isna(col):
        return col

    if isinstance(col, (int, float)):
        if float(col).is_integer():
            return f"fp_{int(col)}"
        return str(col)

    col_str = str(col).strip()

    if col_str.startswith("fp_"):
        return col_str

    try:
        num = float(col_str)
        if num.is_integer():
            return f"fp_{int(num)}"
    except ValueError:
        pass

    return col_str



def read_input_file(input_file: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Read the input file and return a mapping from sheet name to DataFrame.

    Supported formats
    -----------------
    - Excel (.xlsx, .xls): all worksheets are loaded.
    - CSV (.csv): loaded as a single sheet named "Sheet1".
    """
    input_file = str(input_file)
    ext = Path(input_file).suffix.lower()

    if ext in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(input_file)
        return {
            sheet_name: pd.read_excel(input_file, sheet_name=sheet_name)
            for sheet_name in xls.sheet_names
        }

    if ext == ".csv":
        return {"Sheet1": pd.read_csv(input_file)}

    raise ValueError(f"Unsupported file format: {ext}")



def detect_non_fp_columns(df: pd.DataFrame) -> List[str]:
    """
    Return non-fingerprint columns to preserve in the exported result file.

    Typical preserved columns include identifiers and metadata, such as:
    Name, Formula, CAS, RT, and ExactMass.
    """
    return [
        c for c in df.columns
        if not (isinstance(c, str) and c.startswith("fp_"))
    ]


def resolve_name_column(df: pd.DataFrame) -> Optional[str]:
    """
    Resolve the preferred compound-name column for the exported result table.

    The output is restricted to three columns only:
    Name, predicted_pEC50, and status. If no suitable name column is found,
    synthetic names are generated automatically.
    """
    candidates = [
        "Name",
        "name",
        "CompoundName",
        "compound_name",
        "Compound",
        "compound",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None



def infer_model_expected_features(model: object) -> Optional[int]:
    """
    Infer the expected feature dimension from a fitted scikit-learn pipeline.

    For pipelines containing a fitted transformer such as StandardScaler,
    n_features_in_ is used when available.
    """
    try:
        if hasattr(model, "named_steps"):
            for _, step_obj in model.named_steps.items():
                if hasattr(step_obj, "n_features_in_"):
                    return int(step_obj.n_features_in_)
    except Exception:
        pass
    return None



def resolve_exact_mass_column(
    df: pd.DataFrame,
    preferred_col: str = "ExactMass",
) -> Optional[str]:
    """
    Resolve the exact-mass column name.

    The preferred name from the feature configuration is checked first.
    If that column is absent, several common aliases are tested.
    """
    candidates = [
        preferred_col,
        "ExactMass",
        "exact_mass",
        "Exact_Mass",
        "exact mass",
        "Exact mass",
        "MonoisotopicMass",
        "Monoisotopic_Mass",
        "Monoisotopic mass",
        "Monoisotopic Mass",
    ]

    seen = set()
    ordered_candidates: List[str] = []
    for name in candidates:
        if name not in seen:
            ordered_candidates.append(name)
            seen.add(name)

    for col in ordered_candidates:
        if col in df.columns:
            return col

    return None



def load_feature_config(feature_config_path: str | Path) -> Dict[str, object]:
    """
    Load and validate the bundled JSON feature configuration.
    """
    with open(feature_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    fp_order = cfg.get("feature_order", [])
    if not fp_order:
        raise ValueError("feature_order was not found in feature_config.json.")

    include_exact_mass = bool(cfg.get("include_exact_mass", False))
    exact_mass_col = cfg.get("exact_mass_col", "ExactMass")

    model_input_order = cfg.get("model_input_order")
    if not model_input_order:
        model_input_order = fp_order + ([exact_mass_col] if include_exact_mass else [])

    return {
        "model_name": cfg.get("model_name", "SVR_RBF"),
        "target_column": cfg.get("target_column", "pEC50"),
        "feature_prefix": cfg.get("feature_prefix", "fp_"),
        "fp_order": fp_order,
        "n_fp_features": cfg.get("n_fp_features", len(fp_order)),
        "include_exact_mass": include_exact_mass,
        "exact_mass_col": exact_mass_col,
        "model_input_order": model_input_order,
        "n_features_total": cfg.get("n_features_total", len(model_input_order)),
    }



def validate_sheet(
    df: pd.DataFrame,
    fp_order: List[str],
    include_exact_mass: bool = False,
    exact_mass_col: str = "ExactMass",
) -> Dict[str, object]:
    """
    Validate whether one sheet contains the required model inputs.
    """
    fp_cols_now = [c for c in df.columns if isinstance(c, str) and c.startswith("fp_")]
    missing_fp = [fp for fp in fp_order if fp not in fp_cols_now]
    extra_fp = [fp for fp in fp_cols_now if fp not in fp_order]

    actual_exact_mass_col: Optional[str] = None
    exact_mass_missing = False
    exact_mass_invalid_count = 0

    if include_exact_mass:
        actual_exact_mass_col = resolve_exact_mass_column(df, preferred_col=exact_mass_col)
        if actual_exact_mass_col is None:
            exact_mass_missing = True
        else:
            em_values = pd.to_numeric(df[actual_exact_mass_col], errors="coerce")
            exact_mass_invalid_count = int(em_values.isna().sum())

    return {
        "missing_fp": missing_fp,
        "extra_fp": extra_fp,
        "exact_mass_missing": exact_mass_missing,
        "exact_mass_invalid_count": exact_mass_invalid_count,
        "actual_exact_mass_col": actual_exact_mass_col,
    }



def build_model_input(
    df: pd.DataFrame,
    fp_order: List[str],
    include_exact_mass: bool = False,
    exact_mass_col: str = "ExactMass",
) -> np.ndarray:
    """
    Build the model input matrix in the exact order required by the model.
    """
    x_fp = df[fp_order].astype(float)

    if include_exact_mass:
        actual_exact_mass_col = resolve_exact_mass_column(df, preferred_col=exact_mass_col)
        if actual_exact_mass_col is None:
            raise ValueError(f"Required column is missing: {exact_mass_col}")

        em_values = pd.to_numeric(df[actual_exact_mass_col], errors="coerce").astype(float)
        if em_values.isna().any():
            bad_n = int(em_values.isna().sum())
            raise ValueError(
                f"Column '{actual_exact_mass_col}' contains {bad_n} missing or non-numeric values."
            )

        x = pd.concat([x_fp, em_values.rename(exact_mass_col)], axis=1)
    else:
        x = x_fp

    return x.values



def predict_one_sheet(
    df: pd.DataFrame,
    model: object,
    cfg: Dict[str, object],
    sheet_name: str = "Sheet1",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Predict pEC50 values for a single worksheet.
    """
    df = df.copy()
    df.columns = [normalize_fp_name(c) for c in df.columns]

    check = validate_sheet(
        df=df,
        fp_order=cfg["fp_order"],
        include_exact_mass=cfg["include_exact_mass"],
        exact_mass_col=cfg["exact_mass_col"],
    )

    fail_reasons: List[str] = []

    if check["missing_fp"]:
        fail_reasons.append(f"Missing {len(check['missing_fp'])} required fingerprint columns")

    if cfg["include_exact_mass"]:
        if check["exact_mass_missing"]:
            fail_reasons.append(f"Required column is missing: {cfg['exact_mass_col']}")
        elif check["exact_mass_invalid_count"] > 0:
            fail_reasons.append(
                f"Column '{check['actual_exact_mass_col']}' contains "
                f"{check['exact_mass_invalid_count']} missing or non-numeric values"
            )

    name_col = resolve_name_column(df)
    if name_col is not None:
        output_name_series = df[name_col].astype(str)
    else:
        output_name_series = pd.Series(
            [f"Compound_{i + 1}" for i in range(len(df))],
            index=df.index,
            dtype="object",
        )

    if fail_reasons:
        result_df = pd.DataFrame({
            "Name": output_name_series,
            "predicted_pEC50": np.nan,
            "status": "failed",
        })

        summary = {
            "sheet_name": sheet_name,
            "n_rows": len(df),
            "status": "failed",
            "missing_fp_count": len(check["missing_fp"]),
            "extra_fp_count": len(check["extra_fp"]),
            "exact_mass_missing": check["exact_mass_missing"],
            "exact_mass_invalid_count": check["exact_mass_invalid_count"],
            "actual_exact_mass_col": check["actual_exact_mass_col"] or "",
            "missing_fp_example": ", ".join(check["missing_fp"][:10]),
        }
        return result_df, summary

    x = build_model_input(
        df=df,
        fp_order=cfg["fp_order"],
        include_exact_mass=cfg["include_exact_mass"],
        exact_mass_col=cfg["exact_mass_col"],
    )

    expected_n = infer_model_expected_features(model)
    if expected_n is not None and x.shape[1] != expected_n:
        raise ValueError(
            f"The model expects {expected_n} input features, but the current input "
            f"matrix contains {x.shape[1]} features. Please check feature_config.json, "
            "the ExactMass column, and the input template."
        )

    y_pred = model.predict(x)

    result_df = pd.DataFrame({
        "Name": output_name_series,
        "predicted_pEC50": y_pred,
        "status": "success",
    })

    summary = {
        "sheet_name": sheet_name,
        "n_rows": len(df),
        "status": "success",
        "missing_fp_count": 0,
        "extra_fp_count": len(check["extra_fp"]),
        "exact_mass_missing": False,
        "exact_mass_invalid_count": 0,
        "actual_exact_mass_col": check["actual_exact_mass_col"] or "",
        "missing_fp_example": "",
    }

    return result_df, summary



def export_results(
    output_file: str | Path,
    result_dict: Dict[str, pd.DataFrame],
    summary_records: List[Dict[str, object]],
) -> None:
    """
    Export prediction results to one Excel file.

    Each prediction sheet contains exactly three columns:
    Name, predicted_pEC50, and status.
    """
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df_result in result_dict.items():
            safe_sheet_name = str(sheet_name)[:31]
            df_result.to_excel(writer, sheet_name=safe_sheet_name, index=False)

        pd.DataFrame(summary_records).to_excel(writer, sheet_name="Summary", index=False)



def predict_file(
    input_file: str | Path,
    output_file: str | Path,
    model_path: str | Path,
    feature_config_path: str | Path,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """
    Main prediction entry point used by the GUI worker.
    """
    def log(message: str) -> None:
        if logger is not None:
            logger(message)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file was not found: {model_path}")
    if not os.path.exists(feature_config_path):
        raise FileNotFoundError(f"Feature configuration file was not found: {feature_config_path}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file was not found: {input_file}")

    log("Loading the fixed model...")
    model = joblib.load(model_path)
    log("Model loaded successfully.")

    log("Loading the fixed feature configuration...")
    cfg = load_feature_config(feature_config_path)
    log(
        f"Configuration loaded: {cfg['n_fp_features']} fingerprint features"
        + (f" + {cfg['exact_mass_col']}" if cfg["include_exact_mass"] else "")
        + f", total input dimension = {cfg['n_features_total']}."
    )

    expected_n = infer_model_expected_features(model)
    if expected_n is not None:
        log(f"Model internal expected input dimension: {expected_n}")

    log("Reading the input file...")
    sheet_data = read_input_file(input_file)
    log(f"Input file loaded successfully. Detected {len(sheet_data)} worksheet(s).")

    result_dict: Dict[str, pd.DataFrame] = {}
    summary_records: List[Dict[str, object]] = []

    for sheet_name, df in sheet_data.items():
        log(f"Processing worksheet: {sheet_name} | original shape: {df.shape}")

        result_df, summary = predict_one_sheet(
            df=df,
            model=model,
            cfg=cfg,
            sheet_name=sheet_name,
        )

        result_dict[sheet_name] = result_df
        summary_records.append(summary)

        log(
            f"sheet={sheet_name} | status={summary['status']} | rows={summary['n_rows']} | "
            f"missing_fp={summary['missing_fp_count']} | extra_fp={summary['extra_fp_count']} | "
            f"exact_mass_missing={summary['exact_mass_missing']} | "
            f"exact_mass_invalid={summary['exact_mass_invalid_count']}"
        )

    log("Exporting the result file...")
    export_results(output_file, result_dict, summary_records)
    log(f"Result file exported successfully: {Path(output_file).name}")

    return {
        "output_file": str(output_file),
        "summary": pd.DataFrame(summary_records),
    }


# =============================================================================
# GUI worker thread
# =============================================================================
class PredictionWorker(QObject):
    """Background worker for long-running prediction tasks."""

    finished = Signal(dict)
    error = Signal(str)
    log = Signal(str)

    def __init__(self, input_file: str, output_file: str) -> None:
        super().__init__()
        self.input_file = input_file
        self.output_file = output_file

    @Slot()
    def run(self) -> None:
        try:
            result = predict_file(
                input_file=self.input_file,
                output_file=self.output_file,
                model_path=str(MODEL_PATH),
                feature_config_path=str(CONFIG_PATH),
                logger=self.log.emit,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# =============================================================================
# Main window
# =============================================================================
class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.thread: Optional[QThread] = None
        self.worker: Optional[PredictionWorker] = None

        self.setWindowTitle("Batch pEC50 Predictor")
        self.resize(980, 720)

        self._build_ui()
        self._apply_style()
        self._check_internal_resources()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(18)

        # Header card
        header_card = QFrame()
        header_card.setObjectName("HeaderCard")
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(24, 22, 24, 22)
        header_layout.setSpacing(8)

        title = QLabel("Batch pEC50 Predictor")
        title.setObjectName("AppTitle")

        subtitle = QLabel(
            "Import an Excel or CSV file. The application will automatically use the "
            "bundled SVR model and the fixed 60 fingerprint features plus ExactMass "
            "to perform feature selection, ordering, and pEC50 prediction."
        )
        subtitle.setObjectName("AppSubtitle")
        subtitle.setWordWrap(True)

        self.resource_status = QLabel("Internal model status: checking...")
        self.resource_status.setObjectName("ResourceStatus")

        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addWidget(self.resource_status)

        # File card
        file_card = QFrame()
        file_card.setObjectName("Card")
        file_layout = QGridLayout(file_card)
        file_layout.setContentsMargins(22, 20, 22, 20)
        file_layout.setHorizontalSpacing(14)
        file_layout.setVerticalSpacing(14)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Select an Excel or CSV file for prediction")
        self.input_btn = QPushButton("Browse Input File")
        self.input_btn.clicked.connect(self.select_input_file)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Select the export path for the result file")
        self.output_btn = QPushButton("Browse Output Path")
        self.output_btn.clicked.connect(self.select_output_file)

        file_layout.addWidget(QLabel("Input File"), 0, 0)
        file_layout.addWidget(self.input_edit, 0, 1)
        file_layout.addWidget(self.input_btn, 0, 2)

        file_layout.addWidget(QLabel("Output File"), 1, 0)
        file_layout.addWidget(self.output_edit, 1, 1)
        file_layout.addWidget(self.output_btn, 1, 2)

        # Action card
        action_card = QFrame()
        action_card.setObjectName("Card")
        action_layout = QHBoxLayout(action_card)
        action_layout.setContentsMargins(22, 18, 22, 18)
        action_layout.setSpacing(12)

        self.run_btn = QPushButton("Run Prediction")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self.start_prediction)

        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.clicked.connect(self.clear_log)

        action_layout.addWidget(self.run_btn)
        action_layout.addWidget(self.clear_btn)
        action_layout.addStretch()

        # Status card
        status_card = QFrame()
        status_card.setObjectName("Card")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(22, 18, 22, 18)
        status_layout.setSpacing(10)

        self.status_label = QLabel("Current Status: idle")
        self.status_label.setObjectName("SectionLabel")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)

        # Log card
        log_card = QFrame()
        log_card.setObjectName("Card")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(22, 18, 22, 18)
        log_layout.setSpacing(10)

        log_title = QLabel("Execution Log")
        log_title.setObjectName("SectionLabel")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText(
            "The application log will show file loading, ExactMass checks, "
            "fingerprint selection, prediction, and export steps."
        )

        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log_text)

        root.addWidget(header_card)
        root.addWidget(file_card)
        root.addWidget(action_card)
        root.addWidget(status_card)
        root.addWidget(log_card, 1)

        self.statusBar().showMessage("Application started")

    def _apply_style(self) -> None:
        app_font = QFont("Segoe UI", 10)
        QApplication.instance().setFont(app_font)

        self.setStyleSheet(
            """
            QWidget {
                background: #f5f7fb;
                color: #1f2937;
            }

            QMainWindow {
                background: #f5f7fb;
            }

            QFrame#HeaderCard {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1f4e79,
                    stop:1 #2d6ea3
                );
                border-radius: 18px;
            }

            QFrame#Card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 16px;
            }

            QLabel#AppTitle {
                background: transparent;
                color: white;
                font-size: 24px;
                font-weight: 700;
            }

            QLabel#AppSubtitle {
                background: transparent;
                color: rgba(255,255,255,0.92);
                font-size: 12px;
            }

            QLabel#ResourceStatus {
                background: rgba(255,255,255,0.12);
                color: white;
                border-radius: 10px;
                padding: 8px 12px;
                font-size: 12px;
                font-weight: 600;
            }

            QLabel#SectionLabel {
                font-size: 14px;
                font-weight: 700;
                color: #111827;
            }

            QLabel {
                font-size: 13px;
                color: #374151;
                background: transparent;
            }

            QLineEdit {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 10px;
                padding: 10px 12px;
                font-size: 13px;
                selection-background-color: #2d6ea3;
            }

            QLineEdit:focus {
                border: 1px solid #2d6ea3;
            }

            QPushButton {
                background: #eef2f7;
                border: 1px solid #d7dde6;
                border-radius: 10px;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: 600;
                color: #1f2937;
            }

            QPushButton:hover {
                background: #e6ebf3;
            }

            QPushButton:pressed {
                background: #dbe3ee;
            }

            QPushButton#PrimaryButton {
                background: #2d6ea3;
                color: white;
                border: none;
                min-width: 120px;
            }

            QPushButton#PrimaryButton:hover {
                background: #255d89;
            }

            QPushButton#PrimaryButton:pressed {
                background: #1f4e79;
            }

            QPushButton:disabled {
                background: #e5e7eb;
                color: #9ca3af;
                border: 1px solid #e5e7eb;
            }

            QTextEdit {
                background: #0f172a;
                color: #e5eef8;
                border: none;
                border-radius: 12px;
                padding: 12px;
                font-family: Consolas, Segoe UI;
                font-size: 12px;
            }

            QProgressBar {
                background: #edf2f7;
                border: none;
                border-radius: 8px;
                min-height: 12px;
                max-height: 12px;
            }

            QProgressBar::chunk {
                background: #2d6ea3;
                border-radius: 8px;
            }

            QStatusBar {
                background: #ffffff;
                color: #4b5563;
                border-top: 1px solid #e5e7eb;
            }
            """
        )

    def append_log(self, text: str) -> None:
        self.log_text.append(text)
        self.statusBar().showMessage(text)

    def clear_log(self) -> None:
        self.log_text.clear()

    def _check_internal_resources(self) -> None:
        model_exists = MODEL_PATH.exists()
        config_exists = CONFIG_PATH.exists()

        if model_exists and config_exists:
            self.resource_status.setText(
                "Internal model status: fixed SVR model loaded (60 fp + ExactMass)"
            )
            self.append_log(f"[OK] Internal model file: {MODEL_PATH}")
            self.append_log(f"[OK] Internal configuration file: {CONFIG_PATH}")
            self.run_btn.setEnabled(True)
        else:
            messages: List[str] = []
            if not model_exists:
                messages.append(f"Model file is missing: {MODEL_PATH}")
            if not config_exists:
                messages.append(f"Configuration file is missing: {CONFIG_PATH}")

            self.resource_status.setText("Internal model status: missing resources, cannot run")
            for msg in messages:
                self.append_log(f"[ERROR] {msg}")
            self.run_btn.setEnabled(False)

    def select_input_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input File",
            "",
            "Excel/CSV Files (*.xlsx *.xls *.csv);;All Files (*)",
        )
        if file_path:
            self.input_edit.setText(file_path)
            input_path = Path(file_path)
            default_output = input_path.with_name(f"{input_path.stem}_prediction_results.xlsx")
            self.output_edit.setText(str(default_output))

    def select_output_file(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output File",
            "",
            "Excel Files (*.xlsx)",
        )
        if file_path:
            if not file_path.lower().endswith(".xlsx"):
                file_path += ".xlsx"
            self.output_edit.setText(file_path)

    def validate_inputs(self) -> Tuple[str, str]:
        input_file = self.input_edit.text().strip()
        output_file = self.output_edit.text().strip()

        if not input_file:
            raise ValueError("Please select an input file first.")
        if not output_file:
            raise ValueError("Please specify an output file first.")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Internal model file does not exist: {MODEL_PATH}")
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Internal configuration file does not exist: {CONFIG_PATH}")

        return input_file, output_file

    def set_running_state(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.input_btn.setEnabled(not running)
        self.output_btn.setEnabled(not running)

        if running:
            self.progress_bar.setRange(0, 0)
            self.status_label.setText("Current Status: running prediction...")
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)
            self.status_label.setText("Current Status: idle")

    def start_prediction(self) -> None:
        try:
            input_file, output_file = self.validate_inputs()
        except Exception as exc:
            QMessageBox.warning(self, "Incomplete Input", str(exc))
            return

        self.append_log("Preparing to start the prediction task...")
        self.set_running_state(True)

        self.thread = QThread()
        self.worker = PredictionWorker(input_file=input_file, output_file=output_file)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_prediction_finished)
        self.worker.error.connect(self.on_prediction_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @Slot(dict)
    def on_prediction_finished(self, result: Dict[str, object]) -> None:
        self.set_running_state(False)
        output_file = result.get("output_file", "")
        self.append_log("Prediction task completed successfully.")

        QMessageBox.information(
            self,
            "Prediction Completed",
            f"Batch prediction has finished successfully.\n\nResult file:\n{output_file}",
        )

    @Slot(str)
    def on_prediction_error(self, error_message: str) -> None:
        self.set_running_state(False)
        self.append_log(f"[ERROR] {error_message}")

        QMessageBox.critical(
            self,
            "Execution Failed",
            f"An error occurred during prediction:\n\n{error_message}",
        )


# =============================================================================
# Application entry point
# =============================================================================
def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
