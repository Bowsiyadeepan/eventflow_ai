import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ─── RECOMMENDATION ENGINE ───────────────────────────────────────────────────

RECOMMENDATIONS = {
    'vehicle_breakdown': {
        'High': {
            'manpower': '4–6 Traffic Personnel',
            'action': 'Deploy tow truck immediately + lane diversion',
            'barricade': 'Yes – partial road closure',
            'diversion': 'Activate alternate route via parallel road',
            'response_time': '< 10 minutes'
        },
        'Low': {
            'manpower': '1–2 Traffic Personnel',
            'action': 'Assist vehicle to shoulder lane',
            'barricade': 'No',
            'diversion': 'No diversion needed',
            'response_time': '< 20 minutes'
        }
    },
    'accident': {
        'High': {
            'manpower': '6–8 Traffic + Emergency Personnel',
            'action': 'Coordinate with ambulance, clear scene, media control',
            'barricade': 'Yes – full lane closure',
            'diversion': 'Multi-point diversion + public alert',
            'response_time': '< 5 minutes'
        },
        'Low': {
            'manpower': '2–3 Traffic Personnel',
            'action': 'Minor accident clearance + spot fine',
            'barricade': 'Optional',
            'diversion': 'Minor lane shift',
            'response_time': '< 15 minutes'
        }
    },
    'construction': {
        'High': {
            'manpower': '4–5 Traffic Personnel + Site Supervisor',
            'action': 'Enforce construction hours, manage lane discipline',
            'barricade': 'Yes – planned barricading',
            'diversion': 'Pre-announced diversion route',
            'response_time': 'Pre-planned deployment'
        },
        'Low': {
            'manpower': '2 Traffic Personnel',
            'action': 'Monitor and manage slow traffic',
            'barricade': 'Cones only',
            'diversion': 'No diversion needed',
            'response_time': 'Scheduled'
        }
    },
    'public_event': {
        'High': {
            'manpower': '10–15 Traffic Personnel',
            'action': 'Crowd control + event perimeter + parking management',
            'barricade': 'Yes – full perimeter barricading',
            'diversion': 'Full area diversion + public transport reroute',
            'response_time': 'Pre-event deployment'
        },
        'Low': {
            'manpower': '4–6 Traffic Personnel',
            'action': 'Monitor entry/exit points',
            'barricade': 'Partial',
            'diversion': 'Soft diversion advisory',
            'response_time': 'Pre-event deployment'
        }
    },
    'water_logging': {
        'High': {
            'manpower': '3–4 Traffic Personnel + Civic Team',
            'action': 'Road closure + pump deployment + public alert',
            'barricade': 'Yes – full closure',
            'diversion': 'Mandatory diversion to elevated roads',
            'response_time': '< 15 minutes'
        },
        'Low': {
            'manpower': '2 Traffic Personnel',
            'action': 'Monitor water level, advisory to motorists',
            'barricade': 'Warning signs only',
            'diversion': 'Advisory diversion',
            'response_time': '< 30 minutes'
        }
    },
    'pot_holes': {
        'High': {
            'manpower': '2–3 Traffic + Road Maintenance Team',
            'action': 'Emergency patching + speed reduction enforcement',
            'barricade': 'Yes – lane narrowing',
            'diversion': 'Partial lane diversion',
            'response_time': '< 2 hours'
        },
        'Low': {
            'manpower': '1 Traffic Personnel',
            'action': 'Place warning signs, report to PWD',
            'barricade': 'Warning cones',
            'diversion': 'No diversion needed',
            'response_time': 'Next maintenance cycle'
        }
    },
    'congestion': {
        'High': {
            'manpower': '5–8 Traffic Personnel at key junctions',
            'action': 'Manual signal override + junction management',
            'barricade': 'No',
            'diversion': 'Dynamic diversion via VMS boards',
            'response_time': '< 10 minutes'
        },
        'Low': {
            'manpower': '2–3 Traffic Personnel',
            'action': 'Monitor and assist flow',
            'barricade': 'No',
            'diversion': 'Advisory only',
            'response_time': '< 20 minutes'
        }
    },
    'procession': {
        'High': {
            'manpower': '8–12 Traffic Personnel',
            'action': 'Route clearance + crowd escort + junction control',
            'barricade': 'Yes – procession route barricading',
            'diversion': 'Full diversion of cross-traffic',
            'response_time': 'Pre-event deployment'
        },
        'Low': {
            'manpower': '3–4 Traffic Personnel',
            'action': 'Escort and monitor',
            'barricade': 'Minimal',
            'diversion': 'Soft advisory',
            'response_time': 'Pre-event deployment'
        }
    },
    'vip_movement': {
        'High': {
            'manpower': '10+ Traffic Personnel + Pilot Vehicle',
            'action': 'Route clearing, intersection holding, convoy escort',
            'barricade': 'Yes – route barricading',
            'diversion': 'Temporary full route diversion',
            'response_time': 'Pre-planned'
        },
        'Low': {
            'manpower': '4–5 Traffic Personnel',
            'action': 'Pilot + intersection assist',
            'barricade': 'Minimal',
            'diversion': 'Soft advisory',
            'response_time': 'Pre-planned'
        }
    },
    'tree_fall': {
        'High': {
            'manpower': '3–4 Traffic + BBMP Tree Gang',
            'action': 'Road closure + emergency tree removal',
            'barricade': 'Yes – full closure',
            'diversion': 'Mandatory diversion',
            'response_time': '< 15 minutes'
        },
        'Low': {
            'manpower': '2 Traffic Personnel',
            'action': 'Partial clearance + monitor',
            'barricade': 'Partial',
            'diversion': 'Lane shift',
            'response_time': '< 30 minutes'
        }
    },
    'protest': {
        'High': {
            'manpower': '15+ Traffic + Law Enforcement',
            'action': 'Crowd containment + area lockdown',
            'barricade': 'Yes – heavy barricading',
            'diversion': 'Full area diversion',
            'response_time': 'Pre-event or immediate'
        },
        'Low': {
            'manpower': '4–6 Traffic Personnel',
            'action': 'Monitor + maintain flow on periphery',
            'barricade': 'Partial',
            'diversion': 'Soft advisory',
            'response_time': 'Within 30 minutes'
        }
    },
    'others': {
        'High': {
            'manpower': '3–5 Traffic Personnel',
            'action': 'Assess situation on ground, deploy as needed',
            'barricade': 'As required',
            'diversion': 'As required',
            'response_time': '< 15 minutes'
        },
        'Low': {
            'manpower': '1–2 Traffic Personnel',
            'action': 'Monitor and report',
            'barricade': 'No',
            'diversion': 'No',
            'response_time': '< 30 minutes'
        }
    },
    'road_conditions': {
        'High': {
            'manpower': '3–4 Traffic + Maintenance Team',
            'action': 'Speed restriction + repair team deployment',
            'barricade': 'Yes',
            'diversion': 'Lane diversion',
            'response_time': '< 1 hour'
        },
        'Low': {
            'manpower': '1–2 Traffic Personnel',
            'action': 'Warning signs + report to PWD',
            'barricade': 'Cones only',
            'diversion': 'No',
            'response_time': 'Scheduled'
        }
    },
    'Debris': {
        'High': {
            'manpower': '2–3 Traffic + Sanitation Team',
            'action': 'Emergency debris clearance',
            'barricade': 'Yes',
            'diversion': 'Partial diversion',
            'response_time': '< 20 minutes'
        },
        'Low': {
            'manpower': '1 Traffic Personnel',
            'action': 'Minor clearance + report',
            'barricade': 'Cones',
            'diversion': 'No',
            'response_time': '< 45 minutes'
        }
    }
}

def get_recommendation(event_cause, priority):
    cause = event_cause.lower().replace(' ', '_') if event_cause else 'others'
    # match key
    matched = None
    for key in RECOMMENDATIONS:
        if key.lower() == cause:
            matched = key
            break
    if not matched:
        matched = 'others'
    return RECOMMENDATIONS[matched].get(priority, RECOMMENDATIONS[matched]['Low'])


# ─── DATA LOADING & FEATURE ENGINEERING ──────────────────────────────────────

def load_and_engineer(path='astram_events.csv'):
    df = pd.read_csv(path)

    # Parse datetime
    df['start_dt'] = pd.to_datetime(df['start_datetime'], utc=True, errors='coerce')
    df['created_dt'] = pd.to_datetime(df['created_date'], utc=True, errors='coerce')
    df['closed_dt'] = pd.to_datetime(df['closed_datetime'], utc=True, errors='coerce')

    # Duration in minutes
    df['duration_mins'] = (df['closed_dt'] - df['start_dt']).dt.total_seconds() / 60
    df['duration_mins'] = df['duration_mins'].clip(0, 1440)  # cap at 24 hrs

    # Temporal features
    df['hour'] = df['start_dt'].dt.hour
    df['day_of_week'] = df['start_dt'].dt.dayofweek
    df['month'] = df['start_dt'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour'] = df['hour'].apply(lambda h: 1 if (7 <= h <= 10 or 17 <= h <= 21) else 0)
    df['is_night'] = df['hour'].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)

    # Road closure flag
    df['road_closure_flag'] = df['requires_road_closure'].astype(int)

    # Planned vs unplanned
    df['is_planned'] = (df['event_type'] == 'planned').astype(int)

    # Fill missing categoricals
    for col in ['event_cause', 'zone', 'corridor', 'veh_type', 'police_station', 'junction']:
        df[col] = df[col].fillna('Unknown')

    # Label encode categoricals
    le_dict = {}
    for col in ['event_cause', 'zone', 'corridor', 'veh_type', 'police_station']:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    # Target
    df['target'] = (df['priority'] == 'High').astype(int)

    # Has lat/lon
    df['has_location'] = (~df['latitude'].isna()).astype(int)

    return df, le_dict


FEATURE_COLS = [
    'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour', 'is_night',
    'road_closure_flag', 'is_planned',
    'event_cause_enc', 'zone_enc', 'corridor_enc', 'veh_type_enc', 'police_station_enc',
    'has_location', 'duration_mins'
]


# ─── TRAIN ENSEMBLE ──────────────────────────────────────────────────────────

def train_ensemble(df):
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols].fillna(-1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                    num_leaves=31, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05,
                                   max_depth=5, random_state=42,
                                   eval_metric='logloss', verbosity=0)
    xgb_model.fit(X_train, y_train)

    # CatBoost
    cat_model = CatBoostClassifier(iterations=200, learning_rate=0.05,
                                   depth=5, random_state=42, verbose=0)
    cat_model.fit(X_train, y_train)

    # Ensemble probabilities (average)
    lgb_prob  = lgb_model.predict_proba(X_test)[:, 1]
    xgb_prob  = xgb_model.predict_proba(X_test)[:, 1]
    cat_prob  = cat_model.predict_proba(X_test)[:, 1]
    ens_prob  = (lgb_prob + xgb_prob + cat_prob) / 3
    ens_pred  = (ens_prob >= 0.5).astype(int)

    report = classification_report(y_test, ens_pred, output_dict=True)
    acc    = accuracy_score(y_test, ens_pred)

    models = {
        'lgb': lgb_model,
        'xgb': xgb_model,
        'cat': cat_model,
        'features': feat_cols
    }
    return models, report, acc, X_test, y_test


# ─── PREDICT SINGLE EVENT ────────────────────────────────────────────────────

def predict_event(models, le_dict, input_dict):
    """
    input_dict keys: event_cause, zone, corridor, veh_type, police_station,
                     hour, day_of_week, month, is_weekend, is_peak_hour,
                     is_night, road_closure_flag, is_planned, has_location,
                     duration_mins
    """
    row = {}
    for col in ['event_cause', 'zone', 'corridor', 'veh_type', 'police_station']:
        le = le_dict[col]
        val = input_dict.get(col, 'Unknown')
        if val in le.classes_:
            row[col + '_enc'] = le.transform([val])[0]
        else:
            row[col + '_enc'] = 0

    for col in ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
                'is_night', 'road_closure_flag', 'is_planned', 'has_location', 'duration_mins']:
        row[col] = input_dict.get(col, 0)

    feat_cols = models['features']
    X = pd.DataFrame([row])[feat_cols].fillna(-1)

    lgb_p = models['lgb'].predict_proba(X)[0][1]
    xgb_p = models['xgb'].predict_proba(X)[0][1]
    cat_p = models['cat'].predict_proba(X)[0][1]
    avg_p = (lgb_p + xgb_p + cat_p) / 3

    priority = 'High' if avg_p >= 0.5 else 'Low'
    return priority, round(avg_p * 100, 1), {
        'LightGBM': round(lgb_p * 100, 1),
        'XGBoost':  round(xgb_p * 100, 1),
        'CatBoost': round(cat_p * 100, 1)
    }
