import datetime as dt
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html

# --- Read and preprocess CSV ---
csv_file = "detections_log.csv"
df = pd.read_csv(csv_file)

# Convert timestamp to datetime and clean up
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['name', 'timestamp'])
df['confidence'] = df['confidence'].astype(float)
df['detected_duration'] = df['detected_duration'].astype(float)

# --- Extract Day, Week, and Year Automatically ---
df['date'] = df['timestamp'].dt.date
df['week'] = df['timestamp'].dt.isocalendar().week
df['year'] = df['timestamp'].dt.isocalendar().year  # handles year/week overlaps

# --- Determine current date and rolling 7-day window ---
current_date = dt.date.today()
start_of_week = current_date - dt.timedelta(days=6)
current_year = current_date.isocalendar()[0]

# --- Separate datasets ---
# Rolling 7-day window for "current week"
# df_week = df[(df['date'] >= start_of_week) & (df['date'] <= current_date)]
df_week = df.copy()

# All data for the year (for weekly chart)
df_year = df[df['year'] == current_year]

print(f"Current date: {current_date}")
print(f"Showing detections from {start_of_week} to {current_date}")
print(f"Loaded {len(df_week)} detections (last 7 days).")
print(f"Loaded {len(df_year)} total detections this year.")

# --- Professional Color Palette ---
colors = {
    "background": "#F9FAFB",
    "text": "#1F2937",
    "grid": "#E5E7EB",
    "primary": "#2563EB",
    "secondary": "#10B981",
    "accent": "#F59E0B"
}

# --- 1️⃣ 7 Days Visualization (Day by Day with Name Detections) ---
daily_counts = df_week.groupby(['date', 'name']).size().reset_index(name='count')
# Add day names for better readability
daily_counts['day_name'] = pd.to_datetime(daily_counts['date']).dt.strftime('%a %m/%d')

fig1 = px.bar(
    daily_counts,
    x='day_name', y='count',
    color='name',
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Daily Detection Comparison (Last 7 Days)",
    labels={'day_name': 'Day', 'count': 'Number of Detections', 'name': 'Person'},
    text='count'
)
fig1.update_traces(texttemplate='%{text}', textposition='outside')
fig1.update_layout(
    plot_bgcolor=colors["background"],
    paper_bgcolor=colors["background"],
    font_color=colors["text"],
    title_font_size=22,
    xaxis_title="Day of Week",
    yaxis_title="Number of Detections",
    yaxis=dict(gridcolor=colors["grid"]),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        title_text="Person"
    ),
    height=500
)

# --- 2️⃣ Current Date Visualization (Times with Name Detections) ---
df_today = df_week[df_week['date'] == current_date].copy()
if len(df_today) > 0:
    df_today['hour'] = df_today['timestamp'].dt.hour
    df_today['time_label'] = df_today['timestamp'].dt.strftime('%H:%M')
    
    # Create timeline visualization
    fig2 = px.scatter(
        df_today,
        x='timestamp', y='name',
        color='name',
        size='confidence',
        hover_data={'timestamp': '|%H:%M:%S', 'confidence': ':.2f', 'detected_duration': ':.1f'},
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title=f"Today's Detection Timeline ({current_date.strftime('%A, %B %d')})",
        labels={'timestamp': 'Time of Day', 'name': 'Person Detected', 'confidence': 'Confidence %'}
    )
    fig2.update_traces(marker=dict(size=15, line=dict(width=2, color='white')))
    fig2.update_layout(
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
        font_color=colors["text"],
        title_font_size=22,
        xaxis_title="Time of Day",
        yaxis_title="Person",
        yaxis=dict(gridcolor=colors["grid"]),
        xaxis=dict(gridcolor=colors["grid"]),
        height=400
    )
else:
    # Empty chart if no data for today
    fig2 = px.scatter(title=f"Today's Detection Timeline ({current_date.strftime('%A, %B %d')}) - No Data")
    fig2.update_layout(
        plot_bgcolor=colors["background"],
        paper_bgcolor=colors["background"],
        font_color=colors["text"],
        height=400
    )

# --- 3️⃣ Total Detections Confidence Level (Last 7 Days Area Plot) ---
# Calculate confidence by person and date
confidence_by_person = df_week.groupby(['date', 'name'])['confidence'].mean().reset_index()
confidence_by_person['day_name'] = pd.to_datetime(confidence_by_person['date']).dt.strftime('%a %m/%d')

fig3 = px.line(
    confidence_by_person,
    x='day_name',
    y='confidence',
    color='name',
    title="Confidence Level Trend (Last 7 Days)",
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Set1,
    labels={'day_name': 'Day', 'confidence': 'Average Confidence (%)', 'name': 'Person'}
)
fig3.update_traces(mode='lines+markers', line=dict(width=4), marker=dict(size=10))
fig3.update_layout(
    plot_bgcolor=colors["background"],
    paper_bgcolor=colors["background"],
    font_color=colors["text"],
    title_font_size=22,
    xaxis_title="Day of Week",
    yaxis_title="Average Confidence (%)",
    hovermode="x unified",
    yaxis=dict(gridcolor=colors["grid"], range=[0, 100]),
    xaxis=dict(gridcolor=colors["grid"]),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        title_text="Person"
    ),
    height=500
)
# Add reference line at 70% confidence
fig3.add_hline(y=70, line_dash="dash", line_color="gray", 
               annotation_text="Good Confidence (70%)", 
               annotation_position="right")

# --- Create Dash App ---
app = Dash(__name__)
app.title = "Face Recognition Analytics Dashboard"

# --- Summary Statistics ---
total_detections = len(df_week)
unique_people = df_week['name'].nunique()
avg_confidence = df_week['confidence'].mean()
today_detections = len(df_week[df_week['date'] == current_date])

app.layout = html.Div([
    html.H1("Face Recognition Analytics Dashboard",
            style={'textAlign': 'center', 'color': colors["primary"], 'marginBottom': '10px', 'fontWeight': 'bold'}),
    html.P(f"Analysis Period: {start_of_week} to {current_date}",
           style={'textAlign': 'center', 'color': colors["text"], 'fontSize': '16px', 'marginBottom': '30px'}),
    
    # Summary Cards
    html.Div([
        html.Div([
            html.H3(f"{total_detections}", style={'color': colors["primary"], 'margin': '0'}),
            html.P("Total Detections", style={'margin': '5px 0', 'color': colors["text"]})
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': 'white', 
                  'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '22%', 'display': 'inline-block', 'margin': '0 1.5%'}),
        
        html.Div([
            html.H3(f"{unique_people}", style={'color': colors["secondary"], 'margin': '0'}),
            html.P("People Detected", style={'margin': '5px 0', 'color': colors["text"]})
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': 'white',
                  'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '22%', 'display': 'inline-block', 'margin': '0 1.5%'}),
        
        html.Div([
            html.H3(f"{avg_confidence:.1f}%", style={'color': colors["accent"], 'margin': '0'}),
            html.P("Avg Confidence", style={'margin': '5px 0', 'color': colors["text"]})
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': 'white',
                  'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '22%', 'display': 'inline-block', 'margin': '0 1.5%'}),
        
        html.Div([
            html.H3(f"{today_detections}", style={'color': '#8B5CF6', 'margin': '0'}),
            html.P("Today's Detections", style={'margin': '5px 0', 'color': colors["text"]})
        ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': 'white',
                  'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '22%', 'display': 'inline-block', 'margin': '0 1.5%'}),
    ], style={'marginBottom': '30px', 'textAlign': 'center'}),

    # Charts
    html.Div([
        html.Div(dcc.Graph(figure=fig1), style={'width': '100%', 'marginBottom': '20px'}),
        html.Div(dcc.Graph(figure=fig2), style={'width': '100%', 'marginBottom': '20px'}),
        html.Div(dcc.Graph(figure=fig3), style={'width': '100%'}),
    ])
], style={'backgroundColor': colors["background"], 'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

if __name__ == "__main__":
    app.run(debug=True)
