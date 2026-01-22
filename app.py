import streamlit as st
import pickle
import os
import pandas as pd
import json
from train_model import AdvancedColdEmailGenerator
import time
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="AI Powered Cold Email Generator✨",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Playfair+Display:wght@400;700;900&family=Roboto+Mono:wght@400;500;700&family=Inter:wght@300;400;600;800&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 1rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 25px;
        box-shadow: 0 25px 70px rgba(0,0,0,0.4);
        backdrop-filter: blur(10px);
    }
    
    /* 3D Animated Header */
    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(120deg, #667eea, #764ba2, #f093fb, #667eea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: gradientFlow 4s ease infinite, float 3s ease-in-out infinite;
        background-size: 300% 300%;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        letter-spacing: -2px;
    }
    
    @keyframes gradientFlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .sub-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Animated Emoji with 3D effect */
    .animated-emoji {
        display: inline-block;
        animation: bounce3d 2s infinite, rotate3d 4s infinite;
        font-size: 3rem;
        filter: drop-shadow(0 5px 10px rgba(0,0,0,0.3));
    }
    
    @keyframes bounce3d {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-15px) scale(1.1); }
    }
    
    @keyframes rotate3d {
        0%, 100% { transform: rotateY(0deg); }
        50% { transform: rotateY(180deg); }
    }
    
    /* Advanced Form Styles */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        font-family: 'Inter', sans-serif;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 14px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        background: white;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    /* Premium Button Styles */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
        padding: 1rem 2rem;
        border-radius: 50px;
        border: none;
        font-size: 1.15rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover:before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.7);
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    .stDownloadButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        padding: 0.7rem 1.5rem;
        border-radius: 50px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4);
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.6);
    }
    
    /* Premium Email Output Box */
    .email-output {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 2px solid #e0e0e0;
        border-left: 6px solid #667eea;
        font-family: 'Roboto Mono', monospace;
        white-space: pre-wrap;
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        animation: slideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        line-height: 1.9;
        font-size: 0.98rem;
        position: relative;
        overflow: hidden;
    }
    
    .email-output:before {
        content: '📧';
        position: absolute;
        top: 20px;
        right: 20px;
        font-size: 3rem;
        opacity: 0.1;
    }
    
    @keyframes slideIn {
        from { 
            opacity: 0; 
            transform: translateX(30px);
        }
        to { 
            opacity: 1; 
            transform: translateX(0);
        }
    }
    
    /* Section Headers with Icons */
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 12px;
        position: relative;
        padding-bottom: 0.5rem;
    }
    
    .section-header:after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    /* Advanced Metrics */
    .stMetric {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .stMetric label {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        color: #666;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Glass Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.95), rgba(118, 75, 162, 0.95));
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.2);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        background: #f5f5f5;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
    }
    
    /* Feature Box with Animation */
    .feature-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.2), rgba(255,255,255,0.1));
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.7rem 0;
        border-left: 4px solid rgba(255,255,255,0.5);
        font-family: 'Poppins', sans-serif;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .feature-box:hover {
        background: linear-gradient(135deg, rgba(255,255,255,0.3), rgba(255,255,255,0.2));
        transform: translateX(5px);
        border-left-color: white;
    }
    
    /* Divider */
    hr {
        margin: 2.5rem 0;
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-top-color: #667eea !important;
        border-right-color: #764ba2 !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 5px solid #28a745;
        font-family: 'Poppins', sans-serif;
        border-radius: 10px;
        padding: 1.2rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 5px solid #dc3545;
        font-family: 'Poppins', sans-serif;
        border-radius: 10px;
        padding: 1.2rem;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1, #bee5eb);
        border-left: 5px solid #17a2b8;
        font-family: 'Poppins', sans-serif;
        border-radius: 10px;
        padding: 1.2rem;
    }
    
    /* Glass Effect */
    .glass {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Pulse Animation */
    @keyframes pulse {
        0%, 100% { 
            transform: scale(1);
            opacity: 1;
        }
        50% { 
            transform: scale(1.05);
            opacity: 0.8;
        }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generated_email' not in st.session_state:
    st.session_state.generated_email = None
if 'generation_count' not in st.session_state:
    st.session_state.generation_count = 0
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []
if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = {'tones': {}, 'industries': {}, 'lengths': []}

# Load model
@st.cache_resource
def load_model():
    generator = AdvancedColdEmailGenerator()
    
    if os.path.exists('email_generator_model.pkl'):
        generator.load_model()
    else:
        st.warning("🔄 Model not found. Training new model...")
        if os.path.exists('cold_email_dataset.csv'):
            df = pd.read_csv('cold_email_dataset.csv')
            generator.train(df, verbose=False)
            generator.save_model()
            st.success("✅ Model trained successfully!")
        else:
            st.error("❌ Dataset not found. Please run generate_dataset.py first.")
            st.stop()
    
    return generator

# Animated Header
st.markdown("""
    <div class="main-header">
        <span class="animated-emoji">✨</span> AI Powered Cold Email <span class="animated-emoji">🚀</span>
    </div>
    <div class="sub-header">
        Enterprise-Grade Cold Email Generation | Powered by Advanced AI & Machine Learning
    </div>
""", unsafe_allow_html=True)

st.divider()

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📝 Generate Email", "📊 Analytics Dashboard", "⚙️ Advanced Settings"])

with tab1:
    # Main content - full width form
    st.markdown('<div class="section-header">📝 Email Configuration</div>', unsafe_allow_html=True)
    
    # Input form
    with st.form("email_form"):
        st.markdown("### 👤 Recipient Details")
        
        col_a, col_b = st.columns(2)
        with col_a:
            recipient_name = st.text_input(
                "Full Name *",
                placeholder="e.g., Sarah Johnson",
                help="Complete name of your prospect"
            )
        
        with col_b:
            role = st.text_input(
                "Job Title *",
                placeholder="e.g., VP of Engineering",
                help="Current position"
            )
        
        col_c, col_d = st.columns(2)
        with col_c:
            company = st.text_input(
                "Company *",
                placeholder="e.g., TechCorp Inc",
                help="Target organization"
            )
        
        with col_d:
            industry = st.selectbox(
                "Industry",
                options=["", "Technology", "Healthcare", "Finance", "Education", "Retail", 
                        "Manufacturing", "Real Estate", "Marketing", "Logistics", "SaaS"],
                help="Business sector"
            )
        
        st.markdown("### 🎯 Your Offering")
        
        product = st.text_input(
            "Product/Service *",
            placeholder="e.g., AI-Powered Analytics Platform",
            help="Your solution"
        )
        
        pain_point = st.text_input(
            "Pain Point (Optional)",
            placeholder="e.g., Manual data processing inefficiencies",
            help="Specific problem you're solving"
        )
        
        st.markdown("### 🎨 Style & Tone")
        
        col_e, col_f, col_g = st.columns(3)
        
        with col_e:
            tone = st.selectbox(
                "Email Tone *",
                options=["professional", "friendly", "technical", "creative", "consultative"],
                help="Communication style"
            )
        
        with col_f:
            structure = st.selectbox(
                "Structure",
                options=["", "problem_solution", "social_proof", "data_driven", 
                        "storytelling", "question_based"],
                help="Email framework"
            )
        
        with col_g:
            st.markdown("### ")
            tone_emoji = {
                "professional": "💼",
                "friendly": "😊",
                "technical": "🔧",
                "creative": "🎨",
                "consultative": "🤝"
            }
            st.markdown(f"<div style='font-size: 4rem; text-align: center; margin-top: 5px;' class='pulse'>{tone_emoji.get(tone, '✉️')}</div>", unsafe_allow_html=True)
        
        additional_context = st.text_area(
            "Additional Context",
            placeholder="Add specific details, recent news about the company, or personal touches...",
            help="Extra personalization",
            height=100
        )
        
        st.markdown("###")
        submitted = st.form_submit_button("🚀 Generate Premium Email", use_container_width=True)
        
        if submitted:
            if not all([recipient_name, company, role, product]):
                st.error("⚠️ Please complete all required fields marked with *")
            else:
                with st.spinner("✨ Crafting your premium cold email with AI..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    generator = load_model()
                    email = generator.generate_email(
                        recipient_name=recipient_name,
                        company=company,
                        role=role,
                        product=product,
                        tone=tone,
                        industry=industry if industry else None,
                        pain_point=pain_point if pain_point else None,
                        additional_context=additional_context if additional_context else None,
                        structure_preference=structure if structure else None
                    )
                    
                    st.session_state.generated_email = email
                    st.session_state.generation_count += 1
                    
                    # Store analytics
                    st.session_state.generation_history.append({
                        'email': email,
                        'tone': tone,
                        'industry': industry,
                        'timestamp': time.time()
                    })
                    
                    # Update analytics
                    st.session_state.analytics_data['tones'][tone] = st.session_state.analytics_data['tones'].get(tone, 0) + 1
                    if industry:
                        st.session_state.analytics_data['industries'][industry] = st.session_state.analytics_data['industries'].get(industry, 0) + 1
                    st.session_state.analytics_data['lengths'].append(len(email.split()))
                    
                    progress_bar.empty()
                    st.success("✅ Premium email generated successfully!")
                    st.balloons()
    
    # Generated Email Section - Moved to bottom
    if st.session_state.generated_email:
        st.divider()
        st.markdown('<div class="section-header">📧 Generated Email</div>', unsafe_allow_html=True)
        
        st.markdown(
            f'<div class="email-output">{st.session_state.generated_email}</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("###")
        
        # Action buttons
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        
        with col_btn1:
            st.download_button(
                label="📥 Download",
                data=st.session_state.generated_email,
                file_name=f"cold_email_{st.session_state.generation_count}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_btn2:
            if st.button("📋 Copy", use_container_width=True):
                st.code(st.session_state.generated_email, language=None)
                st.info("💡 Select and copy the text above!")
        
        with col_btn3:
            if st.button("🔄 Regenerate", use_container_width=True):
                st.rerun()
        
        with col_btn4:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.generated_email = None
                st.rerun()
        
        # Email Analytics
        st.divider()
        st.markdown('<div class="section-header">📊 Email Insights</div>', unsafe_allow_html=True)
        
        word_count = len(st.session_state.generated_email.split())
        char_count = len(st.session_state.generated_email)
        line_count = len(st.session_state.generated_email.split('\n'))
        sentence_count = st.session_state.generated_email.count('.') + st.session_state.generated_email.count('!') + st.session_state.generated_email.count('?')
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("📝 Words", word_count)
        with stat_col2:
            st.metric("🔤 Characters", char_count)
        with stat_col3:
            st.metric("📄 Lines", line_count)
        with stat_col4:
            st.metric("📋 Sentences", sentence_count)
        
        # Reading metrics
        reading_time = max(1, word_count // 200)
        avg_word_length = char_count // word_count if word_count > 0 else 0
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.info(f"⏱️ **Reading Time:** {reading_time} min")
        with metric_col2:
            st.info(f"📏 **Avg Word Length:** {avg_word_length} chars")

with tab2:
    st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    
    if st.session_state.generation_count > 0:
        col_a1, col_a2, col_a3 = st.columns(3)
        
        with col_a1:
            st.metric("Total Emails Generated", st.session_state.generation_count, "+1" if st.session_state.generation_count > 0 else None)
        with col_a2:
            avg_length = sum(st.session_state.analytics_data['lengths']) / len(st.session_state.analytics_data['lengths']) if st.session_state.analytics_data['lengths'] else 0
            st.metric("Avg Word Count", f"{avg_length:.0f}")
        with col_a3:
            most_used_tone = max(st.session_state.analytics_data['tones'], key=st.session_state.analytics_data['tones'].get) if st.session_state.analytics_data['tones'] else "N/A"
            st.metric("Most Used Tone", most_used_tone.capitalize())
        
        st.divider()
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            if st.session_state.analytics_data['tones']:
                st.markdown("#### 🎭 Tone Distribution")
                fig_tone = px.pie(
                    values=list(st.session_state.analytics_data['tones'].values()),
                    names=list(st.session_state.analytics_data['tones'].keys()),
                    color_discrete_sequence=px.colors.sequential.Purples_r
                )
                fig_tone.update_layout(height=300)
                st.plotly_chart(fig_tone, use_container_width=True)
        
        with col_chart2:
            if st.session_state.analytics_data['industries']:
                st.markdown("#### 🏭 Industry Breakdown")
                fig_industry = px.bar(
                    x=list(st.session_state.analytics_data['industries'].keys()),
                    y=list(st.session_state.analytics_data['industries'].values()),
                    color=list(st.session_state.analytics_data['industries'].values()),
                    color_continuous_scale='Purples'
                )
                fig_industry.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_industry, use_container_width=True)
        
        # Word count trend
        if len(st.session_state.analytics_data['lengths']) > 1:
            st.markdown("#### 📈 Word Count Trend")
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                y=st.session_state.analytics_data['lengths'],
                mode='lines+markers',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8, color='#764ba2')
            ))
            fig_trend.update_layout(
                height=250,
                xaxis_title="Email Number",
                yaxis_title="Word Count",
                showlegend=False
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # Generation History
        st.divider()
        st.markdown("#### 📜 Recent Generation History")
        
        for i, entry in enumerate(reversed(st.session_state.generation_history[-5:])):
            with st.expander(f"Email #{st.session_state.generation_count - i} - {entry['tone'].capitalize()} Tone"):
                st.text(entry['email'][:200] + "..." if len(entry['email']) > 200 else entry['email'])
                st.caption(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp']))}")
    else:
        st.info("📊 Generate some emails to see your analytics dashboard!")

with tab3:
    st.markdown('<div class="section-header">⚙️ Advanced Configuration</div>', unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("#### 🎯 Model Information")
        
        if os.path.exists('email_generator_model.pkl'):
            generator = load_model()
            
            st.markdown("""
                <div class="glass" style="padding: 1.5rem; margin: 1rem 0;">
                    <h4 style="color: white; margin-bottom: 1rem;">✅ Model Status: Active</h4>
                    <p style="color: white; opacity: 0.9;">The AI model is loaded and ready to generate emails.</p>
                </div>
            """, unsafe_allow_html=True)
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("🎨 Available Tones", len(generator.tone_styles))
            with metric_col2:
                st.metric("📝 Template Patterns", len(generator.templates))
            
            if generator.industry_patterns:
                st.metric("🏭 Industry Patterns", len(generator.industry_patterns))
            
            if generator.structure_library:
                st.metric("📋 Email Structures", len(generator.structure_library))
    
    with col_s2:
        st.markdown("#### 🔄 Model Management")
        
        if st.button("🔄 Retrain Model", use_container_width=True):
            if os.path.exists('cold_email_dataset.csv'):
                with st.spinner("🔄 Retraining AI model..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress.progress(i + 1)
                    
                    df = pd.read_csv('cold_email_dataset.csv')
                    generator = AdvancedColdEmailGenerator()
                    generator.train(df, verbose=False)
                    generator.save_model()
                    progress.empty()
                    st.success("✅ Model retrained successfully!")
                    time.sleep(1)
                    st.rerun()
            else:
                st.error("❌ Dataset file not found!")
        
        if st.button("📊 View Model Stats", use_container_width=True):
            if os.path.exists('dataset_metadata.json'):
                with open('dataset_metadata.json', 'r') as f:
                    metadata = json.load(f)
                
                st.json(metadata)
            else:
                st.info("No metadata available")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.generation_history = []
            st.session_state.analytics_data = {'tones': {}, 'industries': {}, 'lengths': []}
            st.success("✅ History cleared!")
            time.sleep(1)
            st.rerun()
    
    st.divider()
    
    # Advanced Features Info
    st.markdown("#### 💡 Advanced Features")
    
    features_info = [
        ("🧠 Neural Pattern Matching", "AI-powered template selection using similarity algorithms"),
        ("🎯 Context-Aware Personalization", "Dynamic content adaptation based on recipient profile"),
        ("🏭 Industry-Specific Optimization", "Tailored language and value propositions per sector"),
        ("📊 Multi-Factor Scoring", "Intelligent ranking of template candidates"),
        ("🔍 Sentiment Analysis", "Tone and sentiment optimization for better engagement"),
        ("📝 Structure Recognition", "Multiple email frameworks for different approaches"),
        ("⚡ Real-Time Generation", "Sub-second email creation with high quality"),
        ("🔒 Enterprise-Grade Security", "Secure data handling and processing")
    ]
    
    for emoji_title, description in features_info:
        st.markdown(f"""
            <div class="feature-box">
                <strong>{emoji_title}</strong><br>
                <small style="opacity: 0.9;">{description}</small>
            </div>
        """, unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("### <span class='pulse'>✨</span> AI Powered Cold Email", unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 📈 Session Statistics", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="glass" style="color: white; text-align: center; padding: 2rem 1rem;">
            <h1 style="margin: 0; font-family: 'Poppins', sans-serif; font-size: 3.5rem; color: white;">{st.session_state.generation_count}</h1>
            <p style="margin: 5px 0; font-family: 'Poppins', sans-serif; opacity: 0.9;">Emails Generated</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("###")
    
    if os.path.exists('email_generator_model.pkl'):
        generator = load_model()
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("🎨 Tones", len(generator.tone_styles), delta=None)
        with col_s2:
            st.metric("📝 Templates", len(generator.templates), delta=None)