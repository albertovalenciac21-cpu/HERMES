from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import FileResponse

from backend.app.ai.model_paths import (
    ModelPathError,
    optimization_report_path,
    optimized_model_path,
    preferred_model_path,
    training_model_path,
    training_report_path,
)
from backend.app.ai.optimizer import AIOptimizationError, optimize_ai_models
from backend.app.ai.predictor import AIPredictionError, predict_existing_patient, predict_from_features
from backend.app.ai.registry import ModelRegistryError, list_registered_models, list_training_reports
from backend.app.ai.report_generator import AIReportError, generate_prediction_pdf
from backend.app.ai.trainer import AITrainingError, train_ai_models

router = APIRouter(prefix="/ai", tags=["Trojan Horse AI"])


@router.post("/train")
def train_trojan_horse_ai(
    dataset_path: str = Query(...),
    model_name: str = Query("tnbc", description="Model namespace, for example brca or tnbc."),
    random_state: int = Query(42, ge=0),
):
    try:
        return train_ai_models(
            dataset_path=dataset_path,
            best_model_path=training_model_path(model_name),
            report_path=training_report_path(model_name),
            random_state=random_state,
            model_name=model_name,
        )
    except (AITrainingError, ModelPathError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/optimize")
def optimize_trojan_horse_ai(
    dataset_path: str = Query(...),
    model_name: str = Query("tnbc"),
    random_state: int = Query(42, ge=0),
):
    try:
        return optimize_ai_models(
            dataset_path=dataset_path,
            optimized_model_path=optimized_model_path(model_name),
            report_path=optimization_report_path(model_name),
            random_state=random_state,
            model_name=model_name,
        )
    except (AIOptimizationError, ModelPathError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/predict")
def predict_with_trojan_horse_ai(
    features: dict[str, Any] = Body(...),
    model_name: str = Query("tnbc"),
    top_feature_count: int = Query(10, ge=1, le=100),
    include_shap: bool = Query(True),
):
    try:
        return predict_from_features(
            features=features,
            model_path=preferred_model_path(model_name),
            model_name=model_name,
            top_feature_count=top_feature_count,
            include_shap=include_shap,
        )
    except (AIPredictionError, ModelPathError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/predict-existing-patient")
def predict_existing_dataset_patient(
    patient_id: str = Query(..., min_length=1),
    model_name: str = Query("tnbc"),
    dataset_path: str | None = Query(None, description="Optional override; defaults to the model package's source dataset."),
    top_feature_count: int = Query(10, ge=1, le=100),
    include_shap: bool = Query(True),
):
    try:
        return predict_existing_patient(
            patient_id=patient_id,
            dataset_path=dataset_path,
            model_path=preferred_model_path(model_name),
            model_name=model_name,
            top_feature_count=top_feature_count,
            include_shap=include_shap,
        )
    except (AIPredictionError, ModelPathError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/report-existing-patient")
def create_existing_patient_report(
    patient_id: str = Query(..., min_length=1),
    model_name: str = Query("tnbc"),
    dataset_path: str | None = Query(None),
    top_feature_count: int = Query(10, ge=1, le=100),
    output_path: str | None = Query(None),
):
    try:
        prediction = predict_existing_patient(
            patient_id=patient_id,
            dataset_path=dataset_path,
            model_path=preferred_model_path(model_name),
            model_name=model_name,
            top_feature_count=top_feature_count,
            include_shap=True,
        )
        report = generate_prediction_pdf(prediction=prediction, output_path=output_path)
        resolved_path = Path(report["report_path"])
        return FileResponse(
            path=str(resolved_path),
            media_type="application/pdf",
            filename=resolved_path.name,
            headers={"X-Research-Use-Only": "true", "X-Patient-ID": patient_id, "X-Model-Name": model_name},
        )
    except (AIPredictionError, AIReportError, ModelPathError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models")
def get_registered_models():
    try:
        return list_registered_models()
    except ModelRegistryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/performance")
def get_training_performance():
    try:
        return list_training_reports()
    except ModelRegistryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
