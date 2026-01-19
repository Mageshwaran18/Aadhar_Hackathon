import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np

# Load the data
file_path = r'D:\Project\Hackathons\Aadhar_Hackathon\high_risk_pincodes_enriched.csv'
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()

print("="*80)
print("HIGH-RISK PINCODES ANALYSIS - EXECUTIVE SUMMARY")
print("="*80)

# Calculate key metrics
total_pincodes = len(df)
total_enrollments = df['Total_Child_Enrollments'].sum()
total_updates = df['Total_Demo_Updates'].sum()
avg_uir = df['UIR'].mean()

print(f"\n📊 KEY METRICS:")
print(f"   Total High-Risk Pincodes: {total_pincodes:,}")
print(f"   Total Child Enrollments: {total_enrollments:,}")
print(f"   Total Demographic Updates: {total_updates:,}")
print(f"   Average Update Rate (UIR): {avg_uir:.2f}%")
print(f"   Coverage Gap: {100 - avg_uir:.2f}%")

# Priority distribution
priority_dist = df['Priority_Level'].value_counts()
print(f"\n🎯 PRIORITY DISTRIBUTION:")
for level, count in priority_dist.items():
    print(f"   {level}: {count} pincodes ({count/total_pincodes*100:.1f}%)")

# State-wise analysis
state_analysis = df.groupby('state').agg({
    'pincode': 'count',
    'Total_Child_Enrollments': 'sum',
    'Total_Demo_Updates': 'sum',
    'UIR': 'mean',
    'Intervention_Priority_Score': 'mean'
}).round(2)
state_analysis.columns = ['Pincodes', 'Enrollments', 'Updates', 'Avg_UIR', 'Avg_Priority_Score']
state_analysis = state_analysis.sort_values('Avg_Priority_Score', ascending=False)

print(f"\n🗺️ TOP 10 STATES BY PRIORITY SCORE:")
print(state_analysis.head(10).to_string())

# Risk factor analysis
risk_factors = {
    'Tribal Areas': df['is_tribal'].sum(),
    'Remote/Rural': df['is_remote_rural'].sum(),
    'Low Literacy': df['is_low_literacy'].sum(),
    'Migration Source': df['is_migration_source'].sum(),
    'Migration Destination': df['is_migration_destination'].sum(),
    'Forest/Hilly': df['is_forest_hilly'].sum()
}

print(f"\n⚠️ RISK FACTOR PREVALENCE:")
for factor, count in risk_factors.items():
    print(f"   {factor}: {count} pincodes ({count/total_pincodes*100:.1f}%)")

# Create visualizations
print("\n" + "="*80)
print("GENERATING INTERACTIVE VISUALIZATIONS...")
print("="*80)

# 1. India Map - State-wise Priority Score
fig1 = go.Figure()

# Prepare state data for choropleth
state_map_data = state_analysis.reset_index()
state_map_data['Pincodes_Text'] = state_map_data['Pincodes'].astype(str)

fig1 = px.choropleth(
    state_map_data,
    geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    featureidkey='properties.ST_NM',
    locations='state',
    color='Avg_Priority_Score',
    color_continuous_scale='Reds',
    hover_data=['Pincodes', 'Enrollments', 'Avg_UIR'],
    title='<b>India Map: State-wise Intervention Priority Scores</b><br><sub>Darker colors indicate higher priority for intervention</sub>',
    labels={'Avg_Priority_Score': 'Priority Score'}
)

fig1.update_geos(
    fitbounds="locations",
    visible=False
)

fig1.update_layout(
    title_font_size=20,
    title_x=0.5,
    geo=dict(
        bgcolor='rgba(0,0,0,0)',
        lakecolor='lightblue',
        landcolor='#f0f0f0'
    ),
    height=700,
    margin={"r":0,"t":80,"l":0,"b":0}
)

# 2. Multi-panel Dashboard
fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Top 10 High-Risk States',
        'Priority Level Distribution',
        'Risk Factor Prevalence',
        'Engagement Level Analysis'
    ),
    specs=[[{"type": "bar"}, {"type": "pie"}],
           [{"type": "bar"}, {"type": "bar"}]]
)

# Top 10 states by priority score
top_10_states = state_analysis.head(10).reset_index()
fig2.add_trace(
    go.Bar(
        x=top_10_states['state'],
        y=top_10_states['Avg_Priority_Score'],
        marker_color='crimson',
        name='Priority Score',
        text=top_10_states['Pincodes'],
        texttemplate='%{text} pincodes',
        textposition='outside'
    ),
    row=1, col=1
)

# Priority distribution pie chart
priority_colors = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
fig2.add_trace(
    go.Pie(
        labels=priority_dist.index,
        values=priority_dist.values,
        marker_colors=[priority_colors.get(x, '#6366f1') for x in priority_dist.index],
        textinfo='label+percent',
        hole=0.3
    ),
    row=1, col=2
)

# Risk factors bar chart
risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Factor', 'Count'])
risk_df['Percentage'] = (risk_df['Count'] / total_pincodes * 100).round(1)
fig2.add_trace(
    go.Bar(
        x=risk_df['Factor'],
        y=risk_df['Percentage'],
        marker_color='#6366f1',
        text=risk_df['Percentage'],
        texttemplate='%{text}%',
        textposition='outside',
        name='Percentage'
    ),
    row=2, col=1
)

# Engagement level analysis
engagement_dist = df['Engagement_Level'].value_counts().sort_index()
fig2.add_trace(
    go.Bar(
        x=engagement_dist.index,
        y=engagement_dist.values,
        marker_color=['#ef4444', '#f59e0b', '#10b981', '#3b82f6'],
        text=engagement_dist.values,
        textposition='outside',
        name='Count'
    ),
    row=2, col=2
)

fig2.update_xaxes(tickangle=-45, row=1, col=1)
fig2.update_xaxes(tickangle=-45, row=2, col=1)
fig2.update_layout(
    title_text="<b>Comprehensive Risk Analysis Dashboard</b>",
    title_font_size=22,
    title_x=0.5,
    showlegend=False,
    height=900
)

# 3. District-level detailed analysis (top 20 districts)
district_analysis = df.groupby(['state', 'district']).agg({
    'pincode': 'count',
    'Total_Child_Enrollments': 'sum',
    'UIR': 'mean',
    'Intervention_Priority_Score': 'mean'
}).round(2)
district_analysis.columns = ['Pincodes', 'Enrollments', 'Avg_UIR', 'Priority_Score']
district_analysis = district_analysis.sort_values('Priority_Score', ascending=False).head(20)
district_analysis['District_State'] = [f"{d[1]}, {d[0]}" for d in district_analysis.index]

fig3 = go.Figure()

fig3.add_trace(go.Bar(
    y=district_analysis['District_State'][::-1],
    x=district_analysis['Priority_Score'][::-1],
    orientation='h',
    marker=dict(
        color=district_analysis['Priority_Score'][::-1],
        colorscale='Reds',
        showscale=True,
        colorbar=dict(title="Priority<br>Score")
    ),
    text=district_analysis['Pincodes'][::-1],
    texttemplate='%{text} pincodes',
    textposition='outside'
))

fig3.update_layout(
    title='<b>Top 20 Districts Requiring Immediate Intervention</b>',
    title_font_size=20,
    title_x=0.5,
    xaxis_title='Intervention Priority Score',
    yaxis_title='',
    height=700,
    margin=dict(l=200)
)

# 4. Scatter plot: UIR vs Enrollments
fig4 = px.scatter(
    df,
    x='Total_Child_Enrollments',
    y='UIR',
    color='Priority_Level',
    size='Intervention_Priority_Score',
    hover_data=['state', 'district', 'pincode'],
    color_discrete_map={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'},
    title='<b>Update Rate vs Child Enrollments by Priority Level</b><br><sub>Bubble size represents intervention priority score</sub>',
    labels={'UIR': 'Update Intensity Rate % ', 'Total_Child_Enrollments': 'Total Child Enrollments'}
)

fig4.update_layout(
    title_font_size=20,
    title_x=0.5,
    height=600
)

# Save all visualizations
print("\n💾 Saving visualizations as HTML files...")
fig1.write_html('india_map_priority_scores.html')
print("   ✓ india_map_priority_scores.html")

fig2.write_html('comprehensive_dashboard.html')
print("   ✓ comprehensive_dashboard.html")

fig3.write_html('top_20_districts.html')
print("   ✓ top_20_districts.html")

fig4.write_html('uir_vs_enrollments_scatter.html')
print("   ✓ uir_vs_enrollments_scatter.html")

# Generate detailed insights report
print("\n" + "="*80)
print("KEY INSIGHTS FOR JUDGES")
print("="*80)

print(f"\n🎯 CRITICAL FINDINGS:")
print(f"\n1. COVERAGE GAP SEVERITY:")
print(f"   • {total_pincodes:,} high-risk pincodes have only {avg_uir:.1f}% average update rate")
print(f"   • This means {100-avg_uir:.1f}% of demographic data is outdated")
print(f"   • {total_enrollments:,} children are affected by this data gap")

high_priority = len(df[df['Priority_Level'] == 'High'])
print(f"\n2. IMMEDIATE ACTION REQUIRED:")
print(f"   • {high_priority} pincodes ({high_priority/total_pincodes*100:.1f}%) need URGENT intervention")
print(f"   • Top 3 states: {', '.join(state_analysis.head(3).index.tolist())}")
print(f"   • These areas should receive 60% of allocated resources")

tribal_count = risk_factors['Tribal Areas']
remote_count = risk_factors['Remote/Rural']
print(f"\n3. GEOGRAPHIC CHALLENGES:")
print(f"   • {tribal_count} ({tribal_count/total_pincodes*100:.1f}%) tribal areas need culturally-sensitive outreach")
print(f"   • {remote_count} ({remote_count/total_pincodes*100:.1f}%) remote areas need mobile enrollment units")
print(f"   • Traditional methods won't work - innovative solutions required")

low_lit = risk_factors['Low Literacy']
print(f"\n4. COMMUNICATION BARRIERS:")
print(f"   • {low_lit} pincodes ({low_lit/total_pincodes*100:.1f}%) have low literacy rates")
print(f"   • Requires: Visual aids, local language support, community leaders")
print(f"   • Digital-first approach will fail in these areas")

migration_total = risk_factors['Migration Source'] + risk_factors['Migration Destination']
print(f"\n5. POPULATION MOBILITY:")
print(f"   • {migration_total} pincodes affected by migration patterns")
print(f"   • Seasonal campaigns timed with migration cycles needed")
print(f"   • Coordination between source and destination areas critical")

print("\n" + "="*80)
print("RECOMMENDED 3-PHASE ACTION PLAN")
print("="*80)

print(f"\n📍 PHASE 1 - IMMEDIATE (0-3 months):")
print(f"   Target: {high_priority} High Priority pincodes")
print(f"   Resources: 60% of budget and personnel")
print(f"   Actions:")
print(f"   • Deploy mobile enrollment units to top 5 states")
print(f"   • Establish temporary centers in tribal/remote areas")
print(f"   • Train local volunteers for community outreach")
print(f"   • Launch awareness campaign in local languages")

medium_priority = len(df[df['Priority_Level'] == 'Medium'])
print(f"\n📍 PHASE 2 - SECONDARY (3-6 months):")
print(f"   Target: {medium_priority} Medium Priority pincodes")
print(f"   Resources: 30% of budget and personnel")
print(f"   Actions:")
print(f"   • Set up semi-permanent enrollment centers")
print(f"   • Leverage schools and anganwadis for outreach")
print(f"   • Digital campaigns in areas with connectivity")
print(f"   • Partner with local NGOs and panchayats")

low_priority = len(df[df['Priority_Level'] == 'Low'])
print(f"\n📍 PHASE 3 - PREVENTIVE (6-12 months):")
print(f"   Target: {low_priority} Low Priority pincodes")
print(f"   Resources: 10% of budget and personnel")
print(f"   Actions:")
print(f"   • Strengthen existing infrastructure")
print(f"   • Implement regular update reminder systems")
print(f"   • Create sustainable community engagement model")
print(f"   • Monitor and prevent future data degradation")

print("\n" + "="*80)
print("SUCCESS METRICS TO TRACK")
print("="*80)
print("\n📈 Quarterly KPIs:")
print("   • Update Rate (UIR) improvement in target pincodes")
print("   • Number of children with updated demographics")
print("   • Coverage of high-risk areas by mobile units")
print("   • Community engagement scores")
print("   • Cost per successful update")
print("   • Reduction in high-priority pincode count")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE - All visualizations saved!")
print("="*80)
print("\nGenerated Files:")
print("   1. india_map_priority_scores.html - Interactive India map")
print("   2. comprehensive_dashboard.html - Multi-panel analytics dashboard")
print("   3. top_20_districts.html - District-level priority ranking")
print("   4. uir_vs_enrollments_scatter.html - Relationship analysis")
print("\nOpen these HTML files in any browser for interactive exploration.")
print("="*80)