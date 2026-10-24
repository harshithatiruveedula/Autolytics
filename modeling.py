"""Automatic, leakage-conscious supervised modelling for AUTOLYTICS."""
from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd

ID_WORDS = re.compile(r'(^|[_\s-])(id|uuid|guid|index|serial|record|patient id|student id|customer id|order id|transaction id|invoice id)([_\s-]|$)', re.I)
LEAK_WORDS = re.compile(r'(target|label|outcome|result|prediction|discharge status|final grade|final score|actual .*|post.?event|after .*|recovery status)', re.I)
NON_FEATURE_WORDS = re.compile(r'(name|email|phone|mobile|address|postcode|zip.?code|notes?|comments?|description|free.?text|url|website|password|token)', re.I)

def get_target_candidates(df: pd.DataFrame) -> list[str]:
    """Return plausible outcome columns, excluding IDs, constants, dates and free text."""
    candidates = []
    if df is None or df.empty:
        return candidates
    for col in df.columns:
        series = df[col]
        nonnull = series.dropna()
        if len(nonnull) < max(20, int(.2 * len(df))):
            continue
        if series.nunique(dropna=True) < 2 or ID_WORDS.search(str(col).strip()):
            continue
        if not pd.api.types.is_numeric_dtype(series) and _is_identifier(series, str(col)):
            continue
        if NON_FEATURE_WORDS.search(str(col)):
            continue
        if pd.api.types.is_datetime64_any_dtype(series) or re.search(r'(^|[_\s-])(date|datetime|timestamp)([_\s-]|$)', str(col), re.I):
            continue
        if pd.api.types.is_numeric_dtype(series):
            # Continuous values are valid regression targets; tiny integer categories are classification targets.
            if series.nunique() > 12 or (series.nunique() >= 2 and not pd.api.types.is_integer_dtype(series)):
                candidates.append(str(col))
            elif series.nunique() >= 2 and series.nunique() <= 12:
                candidates.append(str(col))
        else:
            nunique = nonnull.astype(str).nunique()
            ratio = nunique / max(len(nonnull), 1)
            if nunique <= min(50, max(2, int(.1 * len(nonnull)))) and ratio <= .25:
                candidates.append(str(col))
    return candidates

def _date_expand(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            dt = pd.to_datetime(out[col], errors='coerce')
            out[f'{col}__year'] = dt.dt.year
            out[f'{col}__month'] = dt.dt.month
            out[f'{col}__day'] = dt.dt.day
            out = out.drop(columns=[col])
        elif re.search(r'date|datetime|timestamp', str(col), re.I) and not pd.api.types.is_numeric_dtype(out[col]):
            dt = pd.to_datetime(out[col], errors='coerce')
            if dt.notna().mean() >= .65:
                out[f'{col}__year'] = dt.dt.year
                out[f'{col}__month'] = dt.dt.month
                out[f'{col}__day'] = dt.dt.day
                out = out.drop(columns=[col])
    return out

def _is_identifier(series: pd.Series, name: str) -> bool:
    if ID_WORDS.search(str(name).strip()): return True
    nonnull = series.dropna()
    if len(nonnull) > 0 and nonnull.nunique() / len(nonnull) > .98 and (pd.api.types.is_integer_dtype(series) or pd.api.types.is_object_dtype(series)):
        return True
    return False

def target_task(y: pd.Series):
    y = y.dropna()
    if y.empty: raise ValueError('The selected target contains no usable values.')
    if pd.api.types.is_numeric_dtype(y):
        unique = y.nunique()
        if unique <= 12 or (unique <= 20 and np.allclose(y.astype(float), y.astype(float).round())):
            return 'classification'
        return 'regression'
    if y.nunique() > 100 or y.nunique()/max(len(y),1) > .25:
        raise ValueError('This target has too many distinct text values to learn reliably. Choose a category or numeric outcome instead.')
    return 'classification'

def train_best_model(df: pd.DataFrame, target: str, max_train_rows: int = 18000):
    # Keep scikit-learn out of app startup; load it only after Train Model is clicked.
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    if target not in df.columns: raise ValueError('Choose a valid target column.')
    if len(df) < 20: raise ValueError('At least 20 rows are recommended for a reliable train/test split.')
    y0 = df[target]
    valid = y0.notna()
    work = df.loc[valid].copy()
    y = work.pop(target)
    if _is_identifier(y, target): raise ValueError('This target looks like an identifier, not an outcome. Choose a meaningful result such as sales, score, status, or cost.')
    task = target_task(y)
    if y.nunique() < 2: raise ValueError('The target must contain at least two distinct values.')
    if task == 'classification' and y.value_counts().min() < 2: raise ValueError('Each class needs at least two examples. The selected target is too sparse for a safe split.')
    features=[]
    for c in work.columns:
        if _is_identifier(work[c], c): continue
        if LEAK_WORDS.search(str(c)) and c != target: continue
        if work[c].notna().sum() < max(5, int(.1*len(work))): continue
        if work[c].nunique(dropna=True) <= 1: continue
        features.append(c)
    if not features: raise ValueError('No suitable predictor columns were found after excluding identifiers and unusable fields.')

    # Use only plausible predictors. Exclude identifiers, free-text/admin fields,
    # near-unique columns and obvious post-outcome/leakage columns. This selection
    # uses column metadata only (not target values), so it does not leak test labels.
    useful_features = []
    for col in features:
        if NON_FEATURE_WORDS.search(str(col)):
            continue
        if LEAK_WORDS.search(str(col)):
            continue
        nonnull = work[col].dropna()
        if len(nonnull) and nonnull.nunique() / len(nonnull) > .95:
            continue
        useful_features.append(col)
    # Prefer fields with good coverage and useful variation, without consulting
    # target values or held-out rows. This keeps the prediction form compact.
    useful_features.sort(key=lambda c: (work[c].notna().mean(), min(work[c].nunique(dropna=True), 100) / max(1, work[c].nunique(dropna=True) + 1)), reverse=True)
    features = useful_features[:8]
    if not features:
        raise ValueError('No meaningful predictor columns remain after excluding identifiers, free text and likely leakage fields.')

    # Bound runtime on large uploads: model selection uses a reproducible representative
    # sample, while the final selected pipeline is fitted on a larger bounded sample.
    # This prevents multi-minute browser waits on 50k+ row datasets.
    if len(work) > max_train_rows:
        if task == 'classification':
            try:
                kept_idx, _ = train_test_split(work.index, train_size=max_train_rows, random_state=42, stratify=y)
            except ValueError:
                kept_idx = work.sample(n=max_train_rows, random_state=42).index
        else:
            kept_idx = work.sample(n=max_train_rows, random_state=42).index
        work = work.loc[kept_idx]
        y = y.loc[kept_idx]
    X = _date_expand(work[features])
    # Ensure mixed object columns are handled consistently as categories.
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]): X[c] = X[c].astype('string').astype(object)
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers=[]
    if numeric:
        transformers.append(('numeric', Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]), numeric))
    if categorical:
        transformers.append(('categorical', Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore', sparse_output=True))]), categorical))
    pre=ColumnTransformer(transformers, remainder='drop')
    if task=='classification':
        counts=y.value_counts(); stratify=y if counts.min()>=2 else None
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=stratify)
        models=[('Random Forest',RandomForestClassifier(n_estimators=60,random_state=42,class_weight='balanced_subsample',n_jobs=-1, max_depth=18, min_samples_leaf=2)),('Extra Trees',ExtraTreesClassifier(n_estimators=60,random_state=42,class_weight='balanced',n_jobs=-1, max_depth=18, min_samples_leaf=2)),('Logistic Regression',LogisticRegression(max_iter=1500,class_weight='balanced'))]
        scoring=lambda yt,yp: f1_score(yt,yp,average='weighted',zero_division=0)
        metric='Weighted F1'
    else:
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42)
        models=[('Random Forest',RandomForestRegressor(n_estimators=60,random_state=42,n_jobs=-1, max_depth=18, min_samples_leaf=2)),('Extra Trees',ExtraTreesRegressor(n_estimators=60,random_state=42,n_jobs=-1, max_depth=18, min_samples_leaf=2)),('Ridge Regression',Ridge())]
        scoring=lambda yt,yp: r2_score(yt,yp) if len(yt)>1 else -1
        metric='R²'
    results=[]; fitted=[]
    # A simple mean-prediction baseline is essential for interpreting regression
    # metrics: a complex model should be judged against this held-out baseline.
    baseline_metrics = None
    if task == 'regression':
        baseline_value = float(pd.to_numeric(y_train, errors='coerce').mean())
        baseline_pred = np.full(len(y_test), baseline_value, dtype=float)
        baseline_metrics = {
            'mae': float(mean_absolute_error(y_test, baseline_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
            'r2': float(r2_score(y_test, baseline_pred)) if len(y_test) > 1 else float('nan'),
            'prediction': baseline_value,
        }
    for name, estimator in models:
        pipe=Pipeline([('preprocess',pre),('model',estimator)])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore'); pipe.fit(X_train,y_train); pred=pipe.predict(X_test)
            score=float(scoring(y_test,pred))
            if task == 'classification':
                metrics = {
                    'accuracy': float(accuracy_score(y_test, pred)),
                    'precision': float(precision_score(y_test, pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_test, pred, average='weighted', zero_division=0)),
                    'f1_score': float(f1_score(y_test, pred, average='weighted', zero_division=0)),
                }
            else:
                metrics = {
                    'mae': float(mean_absolute_error(y_test, pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, pred))),
                    'r2': float(r2_score(y_test, pred)) if len(y_test) > 1 else float('nan'),
                }
            results.append({'name':name, 'score':score, 'metric':metric, **metrics})
            fitted.append((score,name,pipe))
        except Exception as exc:
            results.append({'name':name,'error':str(exc)[:220],'metric':metric})
    if not fitted: raise ValueError('All candidate models failed. Check target values and column types.')
    best=max(fitted,key=lambda item:item[0])
    # Refit selected pipeline on the bounded, valid modelling sample. Evaluation above
    # remains held-out; bounded fitting avoids blocking Streamlit for very large files.
    final=best[2]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); final.fit(X,y)
    best_result = next((r for r in results if r.get('name') == best[1]), {})
    predictive_status = 'not_assessed'
    if task == 'regression':
        # Negative R² means the selected model failed to beat the mean baseline
        # on this split; don't present its output as a dependable prediction.
        predictive_status = 'weak' if best_result.get('r2', -1) < 0 else ('moderate' if best_result.get('r2', 0) < .5 else 'stronger')
    extra={'task':task,'target':target,'features':features,'model_name':best[1],'metric':metric,'best_score':best[0], 'results':results,'pipeline':final,'feature_columns':X.columns.tolist(),'feature_template':X.iloc[[0]].copy(),'train_rows':len(X_train),'test_rows':len(X_test),'baseline_metrics':baseline_metrics,'predictive_status':predictive_status,'classes':list(final.named_steps['model'].classes_) if task=='classification' and hasattr(final.named_steps['model'],'classes_') else None}
    return extra

def prepare_prediction(raw: dict, model_info: dict) -> pd.DataFrame:
    row={}
    original_features=model_info['features']
    for col in original_features:
        value=raw.get(col)
        if isinstance(value, str) and value.strip()=='': value=np.nan
        row[col]=value
    X=_date_expand(pd.DataFrame([row]))
    template_cols=model_info['feature_columns']
    for c in template_cols:
        if c not in X: X[c]=np.nan
    return X[template_cols]