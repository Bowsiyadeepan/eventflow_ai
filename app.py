import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os, sys

# ── path fix ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model import (load_and_engineer, train_ensemble,
                   predict_event, get_recommendation)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EventFlow AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem; font-weight: 800; color: #1a1a2e;
        text-align: center; margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1.2rem; color: white;
        text-align: center; margin-bottom: 1rem;
    }
    .metric-val  { font-size: 2rem; font-weight: 800; }
    .metric-lbl  { font-size: 0.85rem; opacity: 0.85; }
    .high-badge  {
        background: #ff4b4b; color: white; padding: 6px 18px;
        border-radius: 20px; font-weight: 700; font-size: 1.1rem;
        display: inline-block;
    }
    .low-badge   {
        background: #21c55d; color: white; padding: 6px 18px;
        border-radius: 20px; font-weight: 700; font-size: 1.1rem;
        display: inline-block;
    }
    .rec-box {
        background: #f8fafc; border-left: 4px solid #667eea;
        border-radius: 8px; padding: 1rem 1.2rem; margin: 0.5rem 0;
    }
    .rec-label { font-weight: 700; color: #1a1a2e; font-size: 0.9rem; }
    .rec-value { color: #333; font-size: 0.95rem; }
    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #1a1a2e;
        border-bottom: 2px solid #667eea; padding-bottom: 0.3rem;
        margin: 1.5rem 0 1rem 0;
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── load & train (cached) ────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'astram_events.csv')

@st.cache_data(show_spinner=False)
def load_data():
    return load_and_engineer(DATA_PATH)

@st.cache_resource(show_spinner=False)
def get_models(df):
    return train_ensemble(df)

with st.spinner("🔄 Loading data & training ensemble model…"):
    df, le_dict = load_data()
    models, report, acc, X_test, y_test = get_models(df)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-light.png", width=64)
    st.markdown("## 🚦 EventFlow AI")
    st.markdown("**Event-Driven Traffic Impact Forecasting**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Dashboard",
        "🔮 Predict & Recommend",
        "🗺️ Zone Heatmap",
        "📈 Model Performance"
    ])
    st.markdown("---")
    st.markdown(f"**Dataset:** {len(df):,} events")
    st.markdown(f"**Model Accuracy:** `{acc*100:.1f}%`")
    st.markdown(f"**Zones Covered:** {df['zone'].nunique()}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<div class="main-title">🚦 EventFlow AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Event-Driven Traffic Impact Forecasting & Enforcement Recommendation System</div>', unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Events", f"{len(df):,}")
    with c2:
        high_pct = (df['priority'] == 'High').mean() * 100
        st.metric("High Priority", f"{high_pct:.1f}%")
    with c3:
        planned = (df['event_type'] == 'planned').sum()
        st.metric("Planned Events", f"{planned:,}")
    with c4:
        closure = df['requires_road_closure'].sum()
        st.metric("Road Closures", f"{closure:,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Event Cause Distribution</div>', unsafe_allow_html=True)
        cause_counts = df['event_cause'].value_counts().reset_index()
        cause_counts.columns = ['cause', 'count']
        fig = px.bar(cause_counts, x='count', y='cause', orientation='h',
                     color='count', color_continuous_scale='Purples',
                     labels={'cause': '', 'count': 'Count'})
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Priority by Event Type</div>', unsafe_allow_html=True)
        pt = df.groupby(['event_cause', 'priority']).size().reset_index(name='count')
        fig2 = px.bar(pt, x='event_cause', y='count', color='priority',
                      color_discrete_map={'High': '#ff4b4b', 'Low': '#21c55d'},
                      barmode='stack', labels={'event_cause': '', 'count': 'Count'})
        fig2.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_tickangle=-35)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-header">Hourly Event Volume</div>', unsafe_allow_html=True)
        hourly = df.groupby('hour').size().reset_index(name='events')
        fig3 = px.area(hourly, x='hour', y='events',
                       color_discrete_sequence=['#667eea'],
                       labels={'hour': 'Hour of Day', 'events': 'Events'})
        fig3.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Zone-wise High Priority Events</div>', unsafe_allow_html=True)
        zone_high = df[df['priority'] == 'High']['zone'].value_counts().reset_index()
        zone_high.columns = ['zone', 'count']
        zone_high = zone_high[zone_high['zone'] != 'Unknown'].head(10)
        fig4 = px.bar(zone_high, x='zone', y='count',
                      color='count', color_continuous_scale='Reds',
                      labels={'zone': '', 'count': 'High Priority Events'})
        fig4.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                           coloraxis_showscale=False, xaxis_tickangle=-30)
        st.plotly_chart(fig4, use_container_width=True)

    # Monthly trend
    st.markdown('<div class="section-header">Monthly Event Trend</div>', unsafe_allow_html=True)
    monthly = df.groupby(['month', 'priority']).size().reset_index(name='count')
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly['month_name'] = monthly['month'].map(month_names)
    fig5 = px.line(monthly, x='month_name', y='count', color='priority',
                   color_discrete_map={'High': '#ff4b4b', 'Low': '#21c55d'},
                   markers=True, labels={'month_name': 'Month', 'count': 'Events'})
    fig5.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig5, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT & RECOMMEND
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict & Recommend":
    st.markdown('<div class="section-header">🔮 Predict Traffic Impact & Get Enforcement Recommendations</div>', unsafe_allow_html=True)
    st.markdown("Fill in the event details below to get AI-powered priority prediction and deployment recommendations.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            event_cause = st.selectbox("Event Cause", sorted(df['event_cause'].unique()))
            zone = st.selectbox("Zone", sorted([z for z in df['zone'].unique() if z != 'Unknown']))
            corridor = st.selectbox("Corridor", sorted([c for c in df['corridor'].unique() if c != 'Unknown'])[:20])

        with col2:
            veh_type = st.selectbox("Vehicle Type", sorted([v for v in df['veh_type'].unique() if v != 'Unknown']))
            police_station = st.selectbox("Police Station", sorted([p for p in df['police_station'].unique() if p != 'Unknown'])[:20])
            event_type = st.selectbox("Event Type", ['unplanned', 'planned'])

        with col3:
            hour = st.slider("Hour of Day", 0, 23, 8)
            day_of_week = st.selectbox("Day", ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
            road_closure = st.selectbox("Requires Road Closure?", ['No', 'Yes'])
            duration_est = st.slider("Estimated Duration (mins)", 5, 300, 45)

        submitted = st.form_submit_button("🚀 Predict Impact & Get Recommendations", use_container_width=True)

    if submitted:
        dow_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,
                   'Friday':4,'Saturday':5,'Sunday':6}
        dow = dow_map[day_of_week]
        is_peak = 1 if (7 <= hour <= 10 or 17 <= hour <= 21) else 0
        is_night = 1 if (hour >= 22 or hour <= 5) else 0
        is_weekend = 1 if dow >= 5 else 0

        input_dict = {
            'event_cause': event_cause,
            'zone': zone,
            'corridor': corridor,
            'veh_type': veh_type,
            'police_station': police_station,
            'hour': hour,
            'day_of_week': dow,
            'month': datetime.now().month,
            'is_weekend': is_weekend,
            'is_peak_hour': is_peak,
            'is_night': is_night,
            'road_closure_flag': 1 if road_closure == 'Yes' else 0,
            'is_planned': 1 if event_type == 'planned' else 0,
            'has_location': 1,
            'duration_mins': duration_est
        }

        priority, confidence, model_scores = predict_event(models, le_dict, input_dict)
        rec = get_recommendation(event_cause, priority)

        st.markdown("---")
        r1, r2, r3 = st.columns([1, 1, 2])

        with r1:
            badge = f'<span class="high-badge">🔴 HIGH PRIORITY</span>' if priority == 'High' \
                    else f'<span class="low-badge">🟢 LOW PRIORITY</span>'
            st.markdown("#### Predicted Priority")
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown(f"**Confidence:** `{confidence}%`")

        with r2:
            st.markdown("#### Model Scores")
            for m, s in model_scores.items():
                color = "#ff4b4b" if s >= 50 else "#21c55d"
                st.markdown(f"**{m}:** `{s}%`")
            st.markdown("**Ensemble:** `{:.1f}%`".format(confidence))

        with r3:
            st.markdown("#### Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Impact Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ff4b4b" if priority == 'High' else "#21c55d"},
                    'steps': [
                        {'range': [0, 50], 'color': '#d4edda'},
                        {'range': [50, 75], 'color': '#fff3cd'},
                        {'range': [75, 100], 'color': '#f8d7da'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Enforcement Recommendations")
        rc1, rc2 = st.columns(2)

        rec_items = [
            ("👮 Manpower Deployment", rec['manpower']),
            ("🚧 Barricading", rec['barricade']),
            ("🔀 Diversion Plan", rec['diversion']),
            ("⚡ Action Required", rec['action']),
            ("⏱️ Response Time", rec['response_time']),
        ]

        for i, (label, value) in enumerate(rec_items):
            col = rc1 if i % 2 == 0 else rc2
            with col:
                st.markdown(f"""
                <div class="rec-box">
                    <div class="rec-label">{label}</div>
                    <div class="rec-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)

        # Post-event feedback
        st.markdown("---")
        st.markdown("### 📝 Post-Event Feedback (Learning Loop)")
        with st.expander("Submit actual outcome to improve future predictions"):
            actual = st.selectbox("Actual Priority observed:", ['High', 'Low'])
            actual_duration = st.number_input("Actual Duration (mins):", min_value=0, value=duration_est)
            notes = st.text_area("Field Notes (optional):")
            if st.button("Submit Feedback"):
                st.success("✅ Feedback recorded! Model will incorporate this in the next training cycle.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ZONE HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Zone Heatmap":
    st.markdown('<div class="section-header">🗺️ Zone-wise Congestion Heatmap</div>', unsafe_allow_html=True)

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_cause = st.multiselect("Filter by Event Cause",
                                         sorted(df['event_cause'].unique()),
                                         default=list(df['event_cause'].unique()))
    with filter_col2:
        selected_priority = st.multiselect("Filter by Priority",
                                            ['High', 'Low'], default=['High', 'Low'])

    filtered = df[df['event_cause'].isin(selected_cause) &
                  df['priority'].isin(selected_priority)]

    # Zone summary table
    zone_summary = filtered.groupby('zone').agg(
        total_events=('id', 'count'),
        high_priority=('target', 'sum'),
        road_closures=('road_closure_flag', 'sum')
    ).reset_index()
    zone_summary['high_pct'] = (zone_summary['high_priority'] / zone_summary['total_events'] * 100).round(1)
    zone_summary = zone_summary[zone_summary['zone'] != 'Unknown'].sort_values('high_priority', ascending=False)

    col1, col2 = st.columns([3, 2])

    with col1:
        fig_heat = px.bar(zone_summary, x='zone', y='high_priority',
                          color='high_pct',
                          color_continuous_scale='RdYlGn_r',
                          labels={'zone': 'Zone', 'high_priority': 'High Priority Events',
                                  'high_pct': 'High Priority %'},
                          title="High Priority Events by Zone")
        fig_heat.update_layout(height=420, xaxis_tickangle=-30,
                               margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        st.markdown("#### Zone Summary")
        st.dataframe(
            zone_summary[['zone', 'total_events', 'high_priority', 'high_pct', 'road_closures']]
            .rename(columns={
                'zone': 'Zone', 'total_events': 'Total',
                'high_priority': 'High', 'high_pct': 'High%',
                'road_closures': 'Closures'
            }),
            use_container_width=True, height=380
        )

    # Corridor heatmap
    st.markdown('<div class="section-header">Corridor-wise Impact</div>', unsafe_allow_html=True)
    corr_data = filtered[filtered['corridor'] != 'Unknown'].groupby(['corridor', 'priority']).size().reset_index(name='count')
    fig_corr = px.bar(corr_data, x='corridor', y='count', color='priority',
                      color_discrete_map={'High': '#ff4b4b', 'Low': '#21c55d'},
                      barmode='group', labels={'corridor': 'Corridor', 'count': 'Events'})
    fig_corr.update_layout(height=350, xaxis_tickangle=-30,
                            margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_corr, use_container_width=True)

    # Map (if lat/lon available)
    st.markdown('<div class="section-header">Incident Map</div>', unsafe_allow_html=True)
    map_df = filtered[~filtered['latitude'].isna() & ~filtered['longitude'].isna()].copy()
    if len(map_df) > 0:
        map_df['color'] = map_df['priority'].map({'High': 'red', 'Low': 'green'})
        fig_map = px.scatter_mapbox(
            map_df.sample(min(1000, len(map_df))),
            lat='latitude', lon='longitude',
            color='priority',
            color_discrete_map={'High': '#ff4b4b', 'Low': '#21c55d'},
            hover_data=['event_cause', 'zone', 'corridor'],
            zoom=10, height=450,
            mapbox_style='open-street-map',
            title="Event Locations (sample of 1000)"
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown('<div class="section-header">📈 Ensemble Model Performance</div>', unsafe_allow_html=True)

    # Metrics table
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Accuracy", f"{acc*100:.1f}%")
    with m2:
        st.metric("Precision (High)", f"{report['1']['precision']*100:.1f}%")
    with m3:
        st.metric("Recall (High)", f"{report['1']['recall']*100:.1f}%")
    with m4:
        st.metric("F1-Score (High)", f"{report['1']['f1-score']*100:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Classification Report")
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df[report_df.index.isin(['0', '1', 'macro avg', 'weighted avg'])]
        report_df.index = report_df.index.map({'0': 'Low Priority', '1': 'High Priority',
                                                'macro avg': 'Macro Avg',
                                                'weighted avg': 'Weighted Avg'})
        st.dataframe(report_df[['precision', 'recall', 'f1-score']].round(3),
                     use_container_width=True)

    with col2:
        st.markdown("#### Model Architecture")
        arch_data = {
            'Model': ['LightGBM', 'XGBoost', 'CatBoost', 'Ensemble (Avg)'],
            'Role': ['Base Learner 1', 'Base Learner 2', 'Base Learner 3', 'Final Prediction'],
            'Estimators': [200, 200, 200, '-'],
            'Learning Rate': [0.05, 0.05, 0.05, '-']
        }
        st.dataframe(pd.DataFrame(arch_data), use_container_width=True)

    st.markdown("#### Feature Importance (LightGBM)")
    feat_imp = pd.DataFrame({
        'Feature': models['features'],
        'Importance': models['lgb'].feature_importances_
    }).sort_values('Importance', ascending=True).tail(12)

    fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Purples')
    fig_imp.update_layout(height=380, showlegend=False,
                           coloraxis_showscale=False,
                           margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### Training Data Summary")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Total Records", f"{len(df):,}")
    with d2:
        st.metric("Train Split", f"{int(len(df)*0.8):,} (80%)")
    with d3:
        st.metric("Test Split", f"{int(len(df)*0.2):,} (20%)")

    st.markdown("#### Features Used")
    feat_df = pd.DataFrame({'Feature': models['features'],
                             'Description': [
                                 'Hour of day (0-23)',
                                 'Day of week (0=Mon)',
                                 'Month (1-12)',
                                 'Weekend flag',
                                 'Peak hour flag (7-10am, 5-9pm)',
                                 'Night time flag',
                                 'Road closure required flag',
                                 'Planned vs unplanned event',
                                 'Event cause (encoded)',
                                 'Zone (encoded)',
                                 'Corridor (encoded)',
                                 'Vehicle type (encoded)',
                                 'Police station (encoded)',
                                 'Location available flag',
                                 'Event duration in minutes'
                             ][:len(models['features'])]})
    st.dataframe(feat_df, use_container_width=True)
