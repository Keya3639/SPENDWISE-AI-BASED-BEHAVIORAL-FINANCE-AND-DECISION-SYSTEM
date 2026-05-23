# =========================================================
# SPENDWISE - FULL UPDATED app.py
# STEP 14 - MULTI AGENT + RESEARCH DASHBOARD
# =========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import tempfile
import os
import calendar

from streamlit_option_menu import option_menu

# =========================================================
# DATABASE
# =========================================================
from utils.database import (
    create_tables,
    add_expense,
    get_expenses
)

# =========================================================
# AI MODULES
# =========================================================
from utils.sentiment import analyze_sentiment
from utils.behavior import detect_behavior
from utils.recommendations import generate_recommendations

# =========================================================
# MEMORY + RAG
# =========================================================
from utils.memory import (
    save_memory,
    load_memory
)

from utils.rag_engine import (
    store_memory,
    retrieve_memory
)

# =========================================================
# AI AGENTS
# =========================================================
from utils.agent import SpendWiseAgent
from utils.autonomous_agent import AutonomousFinancialAgent

# =========================================================
# MULTI AGENT AI
# =========================================================
from utils.multi_agent import (
    savings_agent,
    risk_agent,
    emotion_agent,
    budget_agent,
    investment_agent
)

# =========================================================
# ML PREDICTION ENGINE
# =========================================================
from utils.predictor import (
    predict_future_expense,
    predict_overspending,
    predict_savings,
    financial_health_score
)

# =========================================================
# AI CHAT
# =========================================================
from utils.finance_chat import financial_chat

# =========================================================
# AI SYSTEMS
# =========================================================
from utils.anomaly_detector import detect_anomalies
from utils.goal_planner import generate_savings_plan
from utils.strategy_engine import generate_strategy

# =========================================================
# AI COACH
# =========================================================
from utils.ai_coach import generate_financial_coaching

# =========================================================
# VOICE ASSISTANT FUNCTION
# =========================================================
def speak_text(text):
    """Convert text to speech using gTTS"""
    try:
        if not text:
            return False
        
        if len(text) > 500:
            text = text[:500]
        
        try:
            from gtts import gTTS
        except ImportError:
            st.caption("💬 Install gTTS: pip install gtts")
            return False
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            temp_path = tmp_file.name
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_path)
        
        with open(temp_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
        
        audio_html = f"""
        <audio autoplay="true">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        
        st.markdown(audio_html, unsafe_allow_html=True)
        
        try:
            os.unlink(temp_path)
        except:
            pass
        
        return True
        
    except Exception as e:
        st.caption(f"💬 Voice: {str(e)[:50]}")
        return False

# =========================================================
# INITIALIZATION
# =========================================================
create_tables()

agent = SpendWiseAgent()
auto_agent = AutonomousFinancialAgent()

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SpendWise",
    page_icon="💰",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #161B22,
        #0E1117
    );
}

h1, h2, h3, h4 {
    color: white;
}

[data-testid="metric-container"] {
    background: linear-gradient(
        135deg,
        #1E2633,
        #273549
    );
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #2E3B4E;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #00C6FF,
        #0072FF
    );
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,114,255,0.3);
}

/* Calendar Styles */
.calendar-grid {
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 8px;
    margin-top: 20px;
}
.calendar-day {
    background: #1E2633;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    min-height: 80px;
    transition: transform 0.2s;
}
.calendar-day:hover {
    transform: scale(1.02);
    background: #273549;
}
.calendar-weekday {
    background: #00C6FF20;
    color: #00C6FF;
    font-weight: bold;
    padding: 8px;
    text-align: center;
    border-radius: 8px;
}
.day-number {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 8px;
}
.day-amount {
    font-size: 14px;
    color: #00C6FF;
}
.no-spending {
    color: #666;
    font-size: 12px;
}
.high-spending { background: linear-gradient(135deg, #FF416C, #FF4B2B); }
.medium-spending { background: linear-gradient(135deg, #FFA07A, #FF6B6B); }
.low-spending { background: linear-gradient(135deg, #4CA1AF, #2C3E50); }
</style>
""", unsafe_allow_html=True)

# =========================================================
# EMOTION DETECTION
# =========================================================
def detect_emotion_from_text(text):
    """Convert text to emotion category using keyword matching"""
    try:
        text = str(text).lower()
        if not text or text == 'nan' or len(text.strip()) == 0:
            return "Neutral"
        
        emotion_keywords = {
            "Stress": ["stress", "pressure", "anxiety", "worry", "tense", "overwhelmed"],
            "Sadness": ["sad", "cry", "upset", "depressed", "unhappy", "grief", "lonely", "hurt"],
            "Happiness": ["happy", "joy", "excited", "celebration", "party", "enjoy", "great", "wonderful", "fun"],
            "Anger": ["angry", "frustrated", "annoyed", "mad", "irritated", "furious", "hate"],
            "Fear": ["scared", "fear", "panic", "worried", "anxious", "terrified", "nervous"]
        }
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return emotion
        
        return "Neutral"
        
    except Exception:
        return "Neutral"

# =========================================================
# LOAD DATA
# =========================================================
expenses = get_expenses()

if expenses:
    df = pd.DataFrame(
        expenses,
        columns=[
            "ID",
            "Amount",
            "Category",
            "Note",
            "Date"
        ]
    )
    df["Date"] = pd.to_datetime(df["Date"])
else:
    df = pd.DataFrame(
        columns=[
            "ID",
            "Amount",
            "Category",
            "Note",
            "Date"
        ]
    )

# =========================================================
# SPENDING PERSONALITY
# =========================================================
def spending_personality(df):
    if df.empty:
        return "Unknown"
    
    if "Category" not in df.columns:
        return "Balanced"
    
    shopping_total = df[df["Category"] == "Shopping"]["Amount"].sum()
    total = df["Amount"].sum()
    
    if total == 0:
        return "Balanced"
    
    if shopping_total / total > 0.4:
        return "Impulsive Buyer"
    elif shopping_total / total > 0.25:
        return "Mindful Shopper"
    else:
        return "Balanced Financial Personality"

# =========================================================
# RISK SCORE
# =========================================================
def calculate_risk_score(df):
    if df.empty:
        return 100
    
    negative_count = 0
    if "Sentiment" in df.columns:
        negative_count = len(df[df["Sentiment"] == "Negative"])
    
    impulse_count = 0
    if "Behavior" in df.columns:
        impulse_count = len(df[df["Behavior"] == "Impulse Spending"])
    
    risk_score = max(20, 100 - (negative_count * 5 + impulse_count * 10))
    return int(risk_score)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    selected = option_menu(
        menu_title="💰 SpendWise",
        options=[
            "Dashboard",
            "Add Expense",
            "Analytics",
            "Behavioral AI",
            "AI Coach",
            "Multi-Agent AI",
            "Research Dashboard",
            "AI Voice Assistant",
            "Agentic AI",
            "Reports"
        ],
        icons=[
            "house",
            "plus-circle",
            "bar-chart",
            "mind-share",
            "robot",
            "cpu",
            "activity",
            "mic",
            "diagram-3",
            "file-earmark"
        ],
        default_index=0
    )

# =========================================================
# DASHBOARD
# =========================================================
if selected == "Dashboard":
    st.title("💰 SpendWise")
    st.markdown("### Track. Understand. Optimize.")

    if not df.empty:
        total_spending = df["Amount"].sum()
        avg_spending = df["Amount"].mean()
        total_transactions = len(df)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💸 Total Spending", f"₹{total_spending:,.2f}")
        with col2:
            st.metric("📊 Transactions", total_transactions)
        with col3:
            st.metric("📈 Average Expense", f"₹{avg_spending:.2f}")

        st.markdown("---")

        left, right = st.columns(2)

        with left:
            pie_chart = px.pie(
                df,
                names="Category",
                values="Amount",
                hole=0.5,
                title="Spending by Category"
            )
            st.plotly_chart(pie_chart, use_container_width=True)

        with right:
            category_data = df.groupby("Category")["Amount"].sum().reset_index()
            bar_chart = px.bar(
                category_data,
                x="Category",
                y="Amount",
                color="Category",
                title="Category Spending"
            )
            st.plotly_chart(bar_chart, use_container_width=True)

        st.markdown("---")

        try:
            future_expense = predict_future_expense(df)
            overspending_risk = predict_overspending(df)
            savings_prediction = predict_savings(df)
            health_score = financial_health_score(df)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("🔮 Future Expense", f"₹{future_expense}")
            with c2:
                st.metric("⚠ Risk", overspending_risk)
            with c3:
                st.metric("💰 Savings", f"{savings_prediction}%")
            with c4:
                st.metric("🧠 Health", f"{health_score}%")
        except Exception as e:
            st.info("📊 AI predictions loading...")

        st.markdown("---")

        try:
            recommendations = generate_recommendations(df)
            for rec in recommendations:
                st.info(f"💡 {rec}")
        except Exception as e:
            st.info("💡 Track your expenses regularly to get personalized recommendations")
    else:
        st.info("✨ No expenses yet. Add your first expense to see insights!")

# =========================================================
# ADD EXPENSE
# =========================================================
elif selected == "Add Expense":
    st.title("💰 Add Expense")

    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Amount (₹)", min_value=0.0, step=10.0, format="%.2f")
    
    with col2:
        category = st.selectbox(
            "Category",
            [
                "Food", "Travel", "Shopping", "Bills",
                "Entertainment", "Healthcare", "Education", "Other"
            ]
        )
    
    note = st.text_area("Expense Note (optional)", placeholder="e.g., Bought groceries, Paid bills, Emergency expense...")
    note = note if note else "No note provided"

    if note and note != "No note provided":
        emotion = detect_emotion_from_text(note)
        behavior = detect_behavior(note)
        
        col1, col2 = st.columns(2)
        with col1:
            if emotion in ["Stress", "Sadness", "Anger", "Fear"]:
                st.error(f"🧠 Emotion: {emotion}")
            elif emotion in ["Happiness", "Positive"]:
                st.success(f"😊 Emotion: {emotion}")
            else:
                st.info(f"🧠 Emotion: {emotion}")
        
        with col2:
            st.warning(f"📊 Behavior: {behavior}")

    if st.button("💾 Save Expense", use_container_width=True):
        if amount > 0:
            add_expense(amount, category, note)
            
            try:
                memory_text = f"Expense: ₹{amount} | Category: {category} | Note: {note}"
                save_memory({"amount": amount, "category": category, "note": note})
                store_memory(memory_text, f"expense_{len(df)+1}")
            except Exception as e:
                pass
            
            st.success("✅ Expense saved successfully!")
            st.balloons()
        else:
            st.error("❌ Please enter a valid amount")

# =========================================================
# ANALYTICS WITH CALENDAR VIEW
# =========================================================
elif selected == "Analytics":
    st.title("📊 Analytics Dashboard")
    
    if not df.empty:
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📅 Calendar View", "📊 Categories", "💰 Top Expenses"])
        
        # =========================================================
        # TAB 1: TRENDS
        # =========================================================
        with tab1:
            st.subheader("Daily Spending Trend")
            
            daily_data = df.groupby(df["Date"].dt.date)["Amount"].sum().reset_index()
            daily_data.columns = ['Date', 'Amount']
            
            line_chart = px.line(
                daily_data,
                x="Date",
                y="Amount",
                markers=True,
                title="Daily Spending Trend",
                labels={"Amount": "Spending (₹)", "Date": "Date"}
            )
            line_chart.update_traces(line_color='#00C6FF', marker_color='#0072FF')
            st.plotly_chart(line_chart, use_container_width=True)
            
            # Weekly summary
            st.subheader("Weekly Summary")
            df_copy = df.copy()
            df_copy['Week'] = df_copy['Date'].dt.isocalendar().week
            df_copy['Year'] = df_copy['Date'].dt.year
            weekly_data = df_copy.groupby(['Year', 'Week'])['Amount'].agg(['sum', 'mean', 'count']).reset_index()
            weekly_data['Week Range'] = weekly_data.apply(lambda x: f"Week {x['Week']}, {x['Year']}", axis=1)
            weekly_data.columns = ['Year', 'Week', 'Total (₹)', 'Average (₹)', 'Transactions', 'Week Range']
            st.dataframe(weekly_data[['Week Range', 'Total (₹)', 'Average (₹)', 'Transactions']].head(10), 
                        use_container_width=True)
        
        # =========================================================
        # TAB 2: CALENDAR VIEW
        # =========================================================
        with tab2:
            st.subheader("📅 Spending Calendar")
            st.markdown("*Darker colors = higher spending*")
            
            # Prepare calendar data
            calendar_data = df.copy()
            calendar_data['Date'] = pd.to_datetime(calendar_data['Date'])
            calendar_data['Day'] = calendar_data['Date'].dt.day
            calendar_data['Month'] = calendar_data['Date'].dt.month
            calendar_data['Year'] = calendar_data['Date'].dt.year
            calendar_data['Month_Name'] = calendar_data['Date'].dt.strftime('%B')
            
            # Get unique months in data
            available_months = calendar_data['Date'].dt.to_period('M').unique()
            
            # Month selector
            if len(available_months) > 0:
                selected_month = st.selectbox(
                    "Select Month",
                    options=sorted(available_months, reverse=True),
                    format_func=lambda x: x.strftime('%B %Y')
                )
                
                # Filter data for selected month
                month_data = calendar_data[calendar_data['Date'].dt.to_period('M') == selected_month]
                
                # Create daily spending dictionary
                daily_spending = month_data.groupby(month_data['Date'].dt.day)['Amount'].sum().to_dict()
                
                # Get calendar for the month
                year = selected_month.year
                month = selected_month.month
                cal = calendar.monthcalendar(year, month)
                
                st.markdown(f"### {calendar.month_name[month]} {year}")
                
                # Calendar CSS
                st.markdown("""
                <style>
                .calendar-grid {
                    display: grid;
                    grid-template-columns: repeat(7, 1fr);
                    gap: 8px;
                    margin-top: 20px;
                }
                .calendar-day {
                    background: #1E2633;
                    border-radius: 10px;
                    padding: 12px;
                    text-align: center;
                    min-height: 80px;
                    transition: transform 0.2s;
                }
                .calendar-day:hover {
                    transform: scale(1.02);
                    background: #273549;
                }
                .calendar-weekday {
                    background: #00C6FF20;
                    color: #00C6FF;
                    font-weight: bold;
                    padding: 8px;
                    text-align: center;
                    border-radius: 8px;
                }
                .day-number {
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 8px;
                }
                .day-amount {
                    font-size: 14px;
                    color: #00C6FF;
                }
                .high-spending { background: linear-gradient(135deg, #FF416C, #FF4B2B); }
                .medium-spending { background: linear-gradient(135deg, #FFA07A, #FF6B6B); }
                .low-spending { background: linear-gradient(135deg, #4CA1AF, #2C3E50); }
                </style>
                """, unsafe_allow_html=True)
                
                # Weekday headers
                weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                cols = st.columns(7)
                for i, day in enumerate(weekdays):
                    with cols[i]:
                        st.markdown(f"<div class='calendar-weekday'>{day}</div>", unsafe_allow_html=True)
                
                # Display calendar days
                max_spending = max(daily_spending.values()) if daily_spending else 1
                for week in cal:
                    cols = st.columns(7)
                    for i, day in enumerate(week):
                        with cols[i]:
                            if day == 0:
                                st.markdown("<div class='calendar-day' style='opacity:0.3;'><div class='day-number'> </div></div>", unsafe_allow_html=True)
                            else:
                                amount = daily_spending.get(day, 0)
                                
                                if amount == 0:
                                    color_class = ""
                                    amount_text = "No spending"
                                elif amount < max_spending * 0.3:
                                    color_class = "low-spending"
                                    amount_text = f"₹{amount:,.0f}"
                                elif amount < max_spending * 0.7:
                                    color_class = "medium-spending"
                                    amount_text = f"₹{amount:,.0f}"
                                else:
                                    color_class = "high-spending"
                                    amount_text = f"₹{amount:,.0f}"
                                
                                st.markdown(f"""
                                <div class='calendar-day {color_class}'>
                                    <div class='day-number'>{day}</div>
                                    <div class='day-amount'>{amount_text}</div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Summary stats for the month
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_month = month_data['Amount'].sum()
                    st.metric("📊 Month Total", f"₹{total_month:,.2f}")
                with col2:
                    avg_day = month_data['Amount'].mean() if len(month_data) > 0 else 0
                    st.metric("📈 Daily Avg", f"₹{avg_day:.2f}")
                with col3:
                    days_with_spending = len(daily_spending)
                    total_days = sum(1 for week in cal for day in week if day != 0)
                    st.metric("📆 Active Days", f"{days_with_spending}/{total_days}")
                with col4:
                    highest_day = max(daily_spending.values()) if daily_spending else 0
                    st.metric("🔴 Highest Day", f"₹{highest_day:,.2f}")
            
            # Alternative: Plotly calendar heatmap
            st.markdown("---")
            st.subheader("📊 Calendar Heatmap (Plotly)")
            
            calendar_heatmap_data = df.groupby(df['Date'].dt.date)['Amount'].sum().reset_index()
            calendar_heatmap_data.columns = ['Date', 'Amount']
            calendar_heatmap_data['Date'] = pd.to_datetime(calendar_heatmap_data['Date'])
            
            calendar_heatmap = px.density_heatmap(
                calendar_heatmap_data,
                x=calendar_heatmap_data['Date'].dt.dayofweek,
                y=calendar_heatmap_data['Date'].dt.isocalendar().week,
                z='Amount',
                title="Spending Heatmap (Lighter = Higher Spending)",
                labels={'x': 'Day of Week (0=Monday, 6=Sunday)', 'y': 'Week Number', 'z': 'Amount (₹)'},
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(calendar_heatmap, use_container_width=True)
        
        # =========================================================
        # TAB 3: CATEGORIES
        # =========================================================
        with tab3:
            st.subheader("📊 Category Breakdown")
            
            category_summary = df.groupby("Category")["Amount"].agg(['sum', 'count', 'mean']).reset_index()
            category_summary.columns = ['Category', 'Total (₹)', 'Transactions', 'Average (₹)']
            category_summary['Total (₹)'] = category_summary['Total (₹)'].round(2)
            category_summary['Average (₹)'] = category_summary['Average (₹)'].round(2)
            st.dataframe(category_summary, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                pie_chart = px.pie(
                    category_summary,
                    values='Total (₹)',
                    names='Category',
                    title="Spending by Category",
                    hole=0.3
                )
                st.plotly_chart(pie_chart, use_container_width=True)
            
            with col2:
                bar_chart = px.bar(
                    category_summary,
                    x='Category',
                    y='Total (₹)',
                    color='Category',
                    title="Category Totals"
                )
                st.plotly_chart(bar_chart, use_container_width=True)
        
        # =========================================================
        # TAB 4: TOP EXPENSES
        # =========================================================
        with tab4:
            st.subheader("💰 Top Expenses Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Largest Expenses")
                top_expenses = df.nlargest(10, 'Amount')[['Date', 'Category', 'Amount', 'Note']]
                top_expenses['Date'] = top_expenses['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(top_expenses, use_container_width=True)
            
            with col2:
                st.markdown("### Most Frequent Categories")
                freq_categories = df['Category'].value_counts().reset_index()
                freq_categories.columns = ['Category', 'Count']
                bar_freq = px.bar(
                    freq_categories,
                    x='Category',
                    y='Count',
                    color='Category',
                    title="Transaction Frequency by Category"
                )
                st.plotly_chart(bar_freq, use_container_width=True)
            
            st.markdown("---")
            st.subheader("💡 Spending Insights")
            
            total_spending = df['Amount'].sum()
            avg_expense = df['Amount'].mean()
            top_category = category_summary.loc[category_summary['Total (₹)'].idxmax(), 'Category']
            top_percent = (category_summary['Total (₹)'].max() / total_spending * 100) if total_spending > 0 else 0
            
            insights = []
            if top_percent > 40:
                insights.append(f"⚠️ Your spending is heavily concentrated in **{top_category}** ({top_percent:.0f}% of total). Consider diversifying or reducing in this area.")
            else:
                insights.append(f"✅ Your spending is well-distributed across categories. Keep up the balanced approach!")
            
            if avg_expense > 1000:
                insights.append(f"💰 Your average expense (₹{avg_expense:.0f}) is quite high. Consider breaking down large purchases into planned expenses.")
            elif avg_expense < 300:
                insights.append(f"🎯 Great job keeping expenses small! Your average is just ₹{avg_expense:.0f}.")
            
            if len(df) > 30:
                insights.append(f"📊 You have {len(df)} transactions - excellent tracking habit! This data helps provide better insights.")
            
            for insight in insights:
                st.info(insight)
            
    else:
        st.info("No data available. Add some expenses to see analytics!")

# =========================================================
# BEHAVIORAL AI
# =========================================================
elif selected == "Behavioral AI":
    st.title("🧠 Behavioral AI")

    if not df.empty:
        def safe_behavior(x):
            try:
                return detect_behavior(str(x))
            except:
                return "Normal"
        
        def safe_emotion(x):
            try:
                return detect_emotion_from_text(str(x))
            except:
                return "Neutral"
        
        df["Behavior"] = df["Note"].apply(safe_behavior)
        df["Emotion"] = df["Note"].apply(safe_emotion)

        risk_score = calculate_risk_score(df)
        personality = spending_personality(df)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Risk Score", f"{risk_score}%")
        with col2:
            st.metric("⭐ Personality", personality)
        
        st.markdown("---")
        
        st.subheader("Recent Spending Behavior")
        display_df = df[['Date', 'Category', 'Amount', 'Emotion', 'Behavior']].head(10)
        st.dataframe(display_df, use_container_width=True)
        
        st.subheader("Behavior Distribution")
        behavior_counts = df['Behavior'].value_counts()
        if not behavior_counts.empty:
            fig = px.bar(x=behavior_counts.index, y=behavior_counts.values, title="Spending Behaviors")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        
# =========================================================
# AI COACH
# =========================================================
elif selected == "AI Coach":
    st.title("🤖 Financial AI Coach")

    if not df.empty:
        latest_expense = df.iloc[-1]
        latest_amount = latest_expense["Amount"]
        latest_category = latest_expense["Category"]
        latest_note = str(latest_expense["Note"]) if pd.notna(latest_expense["Note"]) else ""

        detected_emotion = detect_emotion_from_text(latest_note)
        detected_behavior = detect_behavior(latest_note) if latest_note else "Normal"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"💰 Amount: ₹{latest_amount:.2f}")
        with col2:
            st.info(f"📂 Category: {latest_category}")
        with col3:
            if detected_emotion in ["Stress", "Sadness", "Anger", "Fear"]:
                st.error(f"🧠 Emotion: {detected_emotion}")
            elif detected_emotion in ["Happiness", "Positive"]:
                st.success(f"😊 Emotion: {detected_emotion}")
            else:
                st.info(f"🧠 Emotion: {detected_emotion}")
        
        st.warning(f"📊 Behavior: {detected_behavior}")
        
        if st.button("🎯 Generate Personalized Coaching", use_container_width=True):
            with st.spinner("AI Coach is analyzing..."):
                try:
                    ai_response = generate_financial_coaching(
                        detected_behavior,
                        detected_emotion,
                        latest_category,
                        latest_amount
                    )
                    st.success("💡 AI Coach Says:")
                    st.markdown(f"<div style='background-color: #1E2633; padding: 20px; border-radius: 10px;'>{ai_response}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Add an expense first to receive coaching.")

# =========================================================
# MULTI AGENT AI
# =========================================================
elif selected == "Multi-Agent AI":
    st.title("🤖 Multi-Agent Financial Intelligence")

    if not df.empty:
        total_spending = df["Amount"].sum()
        avg_spending = df["Amount"].mean()
        latest_expense = df.iloc[-1]
        latest_note = str(latest_expense["Note"]) if pd.notna(latest_expense["Note"]) else ""
        
        detected_emotion = detect_emotion_from_text(latest_note)
        detected_behavior = detect_behavior(latest_note) if latest_note else "Normal"

        with st.expander("💰 Savings Agent", expanded=True):
            try:
                st.success(savings_agent(total_spending))
            except:
                st.success(f"🎯 Save 20% of your ₹{total_spending:,.2f} spending")

        with st.expander("⚠️ Risk Agent", expanded=True):
            try:
                st.warning(risk_agent(df))
            except:
                avg = df["Amount"].mean() if not df.empty else 0
                st.warning(f"⚠️ Monitor expenses over ₹{avg:.2f}")

        with st.expander("🧠 Emotion Agent", expanded=True):
            try:
                st.info(emotion_agent(detected_emotion, detected_behavior))
            except:
                st.info(f"🧠 Emotional pattern: {detected_emotion}")

        with st.expander("📊 Budget Agent", expanded=True):
            try:
                st.success(budget_agent(avg_spending))
            except:
                st.success(f"📊 Average expense: ₹{avg_spending:.2f}")

        with st.expander("📈 Investment Agent", expanded=True):
            try:
                st.info(investment_agent(total_spending))
            except:
                st.info(f"📈 Invest ₹{total_spending * 0.1:.2f} monthly")
    else:
        st.info("Add expenses to see multi-agent analysis.")

# =========================================================
# RESEARCH DASHBOARD
# =========================================================
elif selected == "Research Dashboard":
    st.title("📊 Research Dashboard")

    if not df.empty:
        def safe_emotion(x):
            try:
                return detect_emotion_from_text(str(x))
            except:
                return "Neutral"
        
        def safe_behavior(x):
            try:
                return detect_behavior(str(x))
            except:
                return "Normal"
        
        df["Emotion"] = df["Note"].apply(safe_emotion)
        df["Behavior"] = df["Note"].apply(safe_behavior)

        heatmap_data = df.groupby([df["Date"].dt.date, "Category"])["Amount"].sum().reset_index()
        if not heatmap_data.empty:
            heatmap = px.density_heatmap(
                heatmap_data,
                x="Date",
                y="Category",
                z="Amount",
                title="Spending Heatmap"
            )
            st.plotly_chart(heatmap, use_container_width=True)

        st.markdown("---")

        emotion_data = df.groupby("Emotion").size().reset_index(name="Count")
        if not emotion_data.empty:
            emotion_chart = px.bar(
                emotion_data,
                x="Emotion",
                y="Count",
                color="Emotion",
                title="Emotion Distribution"
            )
            st.plotly_chart(emotion_chart, use_container_width=True)

        st.markdown("---")

        burst_threshold = df["Amount"].mean() * 2 if not df.empty else 0
        bursts = df[df["Amount"] > burst_threshold]
        
        if not bursts.empty:
            st.error(f"⚠️ {len(bursts)} spending bursts detected")
            st.dataframe(bursts[['Date', 'Category', 'Amount', 'Note']], use_container_width=True)
        else:
            st.success("✅ No spending bursts detected")
            
        st.markdown("---")
        
        st.subheader("Behavior Pattern Analysis")
        behavior_data = df.groupby("Behavior").size().reset_index(name="Count")
        if not behavior_data.empty:
            behavior_chart = px.pie(
                behavior_data,
                values="Count",
                names="Behavior",
                title="Behavior Distribution"
            )
            st.plotly_chart(behavior_chart, use_container_width=True)
    else:
        st.info("Add expenses to see research insights.")

# =========================================================
# AI VOICE ASSISTANT
# =========================================================
elif selected == "AI Voice Assistant":
    st.title("🎤 AI Voice Assistant")
    
    st.markdown("""
    <div style='background-color: #1E2633; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <p>🤖 <strong>Voice Assistant Ready!</strong> Ask any financial question and get a spoken response.</p>
    <p>💡 <strong>Note:</strong> First run <code>pip install gtts</code> in your terminal for voice to work.</p>
    </div>
    """, unsafe_allow_html=True)
    
    user_query = st.text_area("💬 Ask a financial question", 
                               placeholder="Examples:\n• Analyze my budget\n• How can I save more money?\n• What are my spending habits?\n• Where am I spending the most?\n• Give me financial advice",
                               height=120)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎤 Generate Response", use_container_width=True):
            if user_query:
                with st.spinner("AI is thinking..."):
                    try:
                        response = financial_chat(user_query, df)
                        st.session_state['last_response'] = response
                        st.success("🤖 AI Response:")
                        st.markdown(f"<div style='background-color: #1E2633; padding: 15px; border-radius: 10px;'>{response}</div>", unsafe_allow_html=True)
                        
                        try:
                            speak_text(response[:500])
                            st.caption("🔊 Audio playing... check your speakers!")
                        except Exception as e:
                            st.caption(f"💬 Install gTTS: pip install gtts")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a question first")
    
    with col2:
        if 'last_response' in st.session_state:
            if st.button("🔊 Replay", use_container_width=True):
                try:
                    speak_text(st.session_state['last_response'][:500])
                    st.success("🔊 Speaking...")
                except Exception as e:
                    st.error(f"Voice error: {e}")
    
    st.markdown("---")
    
    with st.expander("💡 Example Questions", expanded=False):
        st.markdown("""
        - **"Analyze my budget"** - Complete budget breakdown
        - **"What are my spending habits?"** - Pattern analysis
        - **"How can I save more money?"** - Saving strategies
        - **"Where am I spending most?"** - Top categories
        - **"Give me financial advice"** - Personalized tips
        - **"Summary of my finances"** - Overall health
        - **"Show my recent transactions"** - Last 5 expenses
        """)

# =========================================================
# AGENTIC AI
# =========================================================
elif selected == "Agentic AI":
    st.title("🤖 Agentic AI - Autonomous Financial Intelligence")
    
    if not df.empty:
        with st.expander("📊 Weekly Summary", expanded=True):
            try:
                summary = auto_agent.generate_weekly_summary(df)
                if isinstance(summary, list):
                    for item in summary:
                        st.info(f"📈 {item}")
                else:
                    st.info(f"📈 {summary}")
            except:
                total = df["Amount"].sum()
                count = len(df)
                st.info(f"💰 Total spending: ₹{total:,.2f} across {count} transactions")
        
        with st.expander("🚨 Proactive Alerts", expanded=True):
            try:
                alerts = auto_agent.proactive_alerts(df)
                if isinstance(alerts, list) and alerts:
                    for alert in alerts:
                        st.warning(f"⚠️ {alert}")
                else:
                    st.success("✅ No urgent alerts")
            except:
                st.success("✅ All spending patterns normal")
        
        with st.expander("🔍 Anomaly Detection", expanded=True):
            try:
                anomalies = detect_anomalies(df)
                if anomalies and len(anomalies) > 0:
                    st.error(f"⚠️ {len(anomalies)} anomalies detected")
                    for anomaly in anomalies[:3]:
                        st.write(f"• {anomaly}")
                else:
                    st.success("✅ No anomalies detected")
            except:
                st.info("🔍 Anomaly detection active")
        
        with st.expander("🎯 Savings Goal Planner", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                target_amount = st.number_input("Target (₹)", min_value=1000, step=1000, value=10000)
            with col2:
                months = st.slider("Months", 1, 24, 6)
            
            if st.button("📅 Generate Plan", use_container_width=True):
                try:
                    plan = generate_savings_plan(target_amount, months)
                    st.success(f"""
                    - **Monthly:** ₹{plan['monthly_target']:,.2f}
                    - **Weekly:** ₹{plan['weekly_target']:,.2f}
                    """)
                except:
                    monthly = target_amount / months
                    st.success(f"Save ₹{monthly:,.2f} per month")
        
        with st.expander("💡 Strategic Recommendations", expanded=True):
            try:
                strategies = generate_strategy(df)
                for strategy in strategies:
                    st.info(f"✨ {strategy}")
            except:
                st.info("✨ Continue tracking for personalized strategies")
    else:
        st.info("📝 Add expenses to activate AI agents")

# =========================================================
# REPORTS
# =========================================================
elif selected == "Reports":
    st.title("📄 Financial Reports")

    if not df.empty:
        total_spending = df["Amount"].sum()
        avg_spending = df["Amount"].mean()
        unique_categories = df["Category"].nunique()
        highest_expense = df["Amount"].max()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Spending", f"₹{total_spending:,.2f}")
        with col2:
            st.metric("Average Expense", f"₹{avg_spending:.2f}")
        with col3:
            st.metric("Categories", unique_categories)
        with col4:
            st.metric("Highest", f"₹{highest_expense:.2f}")
        
        st.markdown("---")
        
        st.subheader("📋 All Transactions")
        display_df = df[['Date', 'Category', 'Amount', 'Note']].copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(display_df, use_container_width=True)
        
        st.subheader("📊 Category Summary")
        category_summary = df.groupby('Category').agg({
            'Amount': ['sum', 'mean', 'count']
        }).round(2)
        category_summary.columns = ['Total (₹)', 'Average (₹)', 'Count']
        st.dataframe(category_summary, use_container_width=True)
        
        st.subheader("📅 Monthly Summary")
        df['Month'] = df['Date'].dt.strftime('%Y-%m')
        monthly_summary = df.groupby('Month').agg({
            'Amount': ['sum', 'mean', 'count']
        }).round(2)
        monthly_summary.columns = ['Total (₹)', 'Average (₹)', 'Transactions']
        st.dataframe(monthly_summary, use_container_width=True)
        
        csv = df[['Date', 'Category', 'Amount', 'Note']].copy()
        csv['Date'] = csv['Date'].dt.strftime('%Y-%m-%d')
        csv = csv.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="spendwise_report.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("No data available. Add expenses to generate reports.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>💡 SpendWise AI - Your Intelligent Financial Assistant</div>",
    unsafe_allow_html=True
)