# import packages
import streamlit as st
import pandas as pd
import re
import altair as alt
import json
from snowflake.snowpark.context import get_active_session

# Helper function to clean text
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

st.set_page_config(page_title="GenAI Reviews App", page_icon="🤖", layout="wide")

st.title("🤖 Hello, GenAI!")
st.write("This is your GenAI-powered data processing app.")

# Get Snowflake session
session = get_active_session()

# Create tabs
tab1, tab2, tab3 = st.tabs(["📤 Data Ingestion", "📊 Data & Plots", "🔍 RAG App"])

# ============================================================================
# TAB 1: Data Ingestion
# ============================================================================
with tab1:
    st.header("📥 Data Ingestion & Processing")
    
    # Layout two buttons side by side
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Ingest Dataset"):
            try:
                # Load from Snowflake stage instead of local file
                query = """
                SELECT * FROM customer_reviews
                LIMIT 1000
                """
                st.session_state["df"] = session.sql(query).to_pandas()
                st.success(f"✅ Dataset loaded successfully! ({len(st.session_state['df']):,} rows)")
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")

    with col2:
        if st.button("🧹 Parse Reviews"):
            if "df" in st.session_state:
                if "SUMMARY" in st.session_state["df"].columns:
                    st.session_state["df"]["CLEANED_SUMMARY"] = st.session_state["df"]["SUMMARY"].apply(clean_text)
                    st.success("✅ Reviews parsed and cleaned!")
                else:
                    st.warning("⚠️ No SUMMARY column found in dataset")
            else:
                st.warning("⚠️ Please ingest the dataset first.")

    # Display the dataset if it exists
    if "df" in st.session_state:
        # Product filter dropdown
        st.subheader("🔍 Filter by Product")
        product = st.selectbox(
            "Choose a product", 
            ["All Products"] + sorted(list(st.session_state["df"]["PRODUCT"].unique())),
            key="tab1_product"
        )
        
        st.subheader(f"📁 Reviews for {product}")

        if product != "All Products":
            filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
        else:
            filtered_df = st.session_state["df"]
        
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        
        # Visualization using Altair (instead of matplotlib)
        if "SENTIMENT_SCORE" in st.session_state["df"].columns:
            st.subheader("📊 Average Sentiment Score by Product")
            
            grouped = st.session_state["df"].groupby("PRODUCT")["SENTIMENT_SCORE"].mean().reset_index()
            grouped = grouped.sort_values("SENTIMENT_SCORE")
            
            chart = alt.Chart(grouped).mark_bar().encode(
                x=alt.X('SENTIMENT_SCORE:Q', title='Average Sentiment Score'),
                y=alt.Y('PRODUCT:N', sort='-x', title='Product'),
                color=alt.Color('SENTIMENT_SCORE:Q', 
                               scale=alt.Scale(scheme='redyellowgreen'),
                               legend=None),
                tooltip=['PRODUCT', alt.Tooltip('SENTIMENT_SCORE:Q', format='.3f')]
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)

# ============================================================================
# TAB 2: Data and Plots
# ============================================================================
with tab2:
    st.title("📊 Customer Sentiment and Delivery Analysis")

    # Data loading functions
    @st.cache_data
    def load_data():
        query_reviews = """
        SELECT *
        FROM REVIEWS_WITH_SENTIMENT
        """
        return session.sql(query_reviews).to_pandas()

    # Load data
    with st.spinner("📥 Loading data..."):
        try:
            df = load_data()
            st.success(f"✅ Loaded {len(df):,} reviews")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.stop()

    # ========================================================================
    # Key Metrics
    # ========================================================================
    st.subheader("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    
    with col2:
        avg_sentiment = df["SENTIMENT_SCORE"].mean()
        st.metric("Avg Sentiment", f"{avg_sentiment:.3f}")
    
    with col3:
        positive_pct = (df["SENTIMENT_SCORE"] >= 0.5).sum() / len(df) * 100
        st.metric("Positive %", f"{positive_pct:.1f}%")
    
    with col4:
        if "LATE" in df.columns:
            late_pct = (df["LATE"] == True).sum() / len(df) * 100
            st.metric("Late Deliveries %", f"{late_pct:.1f}%")

    st.markdown("---")

    # ========================================================================
    # Average sentiment by product (Using Altair)
    # ========================================================================
    st.header("📊 Average Sentiment by Product")
    
    avg_sentiment_product = df.groupby("PRODUCT")["SENTIMENT_SCORE"].mean().reset_index()
    avg_sentiment_product = avg_sentiment_product.sort_values("SENTIMENT_SCORE")

    chart1 = alt.Chart(avg_sentiment_product).mark_bar().encode(
        x=alt.X('SENTIMENT_SCORE:Q', title='Average Sentiment Score', scale=alt.Scale(domain=[-1, 1])),
        y=alt.Y('PRODUCT:N', sort='-x', title='Product'),
        color=alt.Color('SENTIMENT_SCORE:Q', 
                       scale=alt.Scale(scheme='redyellowgreen', domain=[-1, 1]),
                       legend=None),
        tooltip=['PRODUCT', alt.Tooltip('SENTIMENT_SCORE:Q', format='.3f')]
    ).properties(height=500)

    st.altair_chart(chart1, use_container_width=True)

    # ========================================================================
    # Filter by product selection
    # ========================================================================
    st.markdown("---")
    st.subheader("🔍 Filter by Product")
    
    product = st.selectbox(
        "Choose a product", 
        ["All Products"] + sorted(list(df["PRODUCT"].unique())),
        key="tab2_product"
    )

    if product != "All Products":
        filtered_data = df[df["PRODUCT"] == product]
    else:
        filtered_data = df

    # Display filtered dataset
    st.subheader(f"📁 Reviews for {product}")
    st.dataframe(filtered_data, use_container_width=True, hide_index=True, height=400)

    # ========================================================================
    # Average sentiment by delivery status (Using Altair)
    # ========================================================================
    if "STATUS" in filtered_data.columns:
        st.header(f"📦 Average Sentiment by Delivery Status for {product}")
        
        avg_sentiment_status = filtered_data.groupby("STATUS")["SENTIMENT_SCORE"].mean().reset_index()
        avg_sentiment_status = avg_sentiment_status.sort_values("SENTIMENT_SCORE")

        chart2 = alt.Chart(avg_sentiment_status).mark_bar(color='steelblue').encode(
            x=alt.X('SENTIMENT_SCORE:Q', title='Average Sentiment Score'),
            y=alt.Y('STATUS:N', sort='-x', title='Delivery Status'),
            tooltip=['STATUS', alt.Tooltip('SENTIMENT_SCORE:Q', format='.3f')]
        ).properties(height=300)

        st.altair_chart(chart2, use_container_width=True)
    
    # ========================================================================
    # Sentiment Distribution
    # ========================================================================
    st.header(f"📊 Sentiment Distribution for {product}")
    
    hist_chart = alt.Chart(filtered_data).mark_bar(color='coral').encode(
        alt.X('SENTIMENT_SCORE:Q', bin=alt.Bin(maxbins=30), title='Sentiment Score'),
        alt.Y('count()', title='Frequency'),
        tooltip=['count()']
    ).properties(height=400)
    
    st.altair_chart(hist_chart, use_container_width=True)

# ============================================================================
# TAB 3: RAG App
# ============================================================================
with tab3:
    st.title("🔍 RAG App - Cortex Search")
    st.markdown("Ask questions about your product reviews using AI-powered search")

    # Input box for user prompt
    prompt = st.text_input(
        "💬 Enter your query:", 
        value="Any goggles review?",
        placeholder="e.g., What do customers say about goggles?"
    )

    if prompt:
        if st.button("🔎 Run Query", type="primary"):
            with st.spinner("🔍 Searching..."):
                try:
                    # Use SQL to query Cortex Search Service
                    search_query = f"""
                    SELECT 
                        value:CHUNK::string as chunk,
                        value:file_name::string as file_name
                    FROM TABLE(
                        AITECHSKILL_DB.AITECHSKILL_SCHEMA.AITECHSKILL_SEARCH_SERVICE(
                            '{prompt.replace("'", "''")}'
                        )
                    )
                    LIMIT 3
                    """
                    
                    search_df = session.sql(search_query).to_pandas()
                    
                    if len(search_df) > 0:
                        st.success(f"✅ Found {len(search_df)} relevant results")
                        
                        for idx, row in search_df.iterrows():
                            with st.container():
                                st.markdown(f"### Result {idx + 1}")
                                st.info(row['CHUNK'])
                                st.caption(f"📄 Source: {row['FILE_NAME']}")
                                st.markdown("---")
                    else:
                        st.warning("No results found. Try a different query.")
                        
                except Exception as e:
                    st.error(f"❌ Error during search: {str(e)}")
                    
                    # Fallback: Try alternative approach
                    st.info("💡 Trying alternative search method...")
                    
                    try:
                        # Alternative: Direct table query with text matching
                        fallback_query = f"""
                        SELECT 
                            REVIEW_TEXT as chunk,
                            PRODUCT as file_name
                        FROM REVIEWS_WITH_SENTIMENT
                        WHERE LOWER(REVIEW_TEXT) LIKE LOWER('%{prompt}%')
                        LIMIT 3
                        """
                        
                        fallback_df = session.sql(fallback_query).to_pandas()
                        
                        if len(fallback_df) > 0:
                            st.success(f"✅ Found {len(fallback_df)} matching reviews")
                            
                            for idx, row in fallback_df.iterrows():
                                with st.container():
                                    st.markdown(f"### Result {idx + 1}")
                                    st.info(row['CHUNK'])
                                    st.caption(f"📦 Product: {row['FILE_NAME']}")
                                    st.markdown("---")
                        else:
                            st.warning("No matching reviews found.")
                            
                    except Exception as e2:
                        st.error(f"❌ Fallback search also failed: {str(e2)}")

    # ========================================================================
    # Additional: AI-Powered Q&A
    # ========================================================================
    st.markdown("---")
    st.subheader("💬 AI-Powered Q&A")
    st.markdown("Ask questions and get AI-generated insights from your reviews")
    
    qa_question = st.text_area(
        "Ask a question about your reviews:",
        placeholder="e.g., What are the main complaints about delivery times?"
    )
    
    if qa_question and st.button("🤖 Get AI Answer", type="secondary"):
        with st.spinner("🤔 Generating answer..."):
            try:
                # Get sample reviews for context
                context_query = """
                SELECT 
                    PRODUCT,
                    REVIEW_TEXT,
                    SENTIMENT_SCORE
                FROM REVIEWS_WITH_SENTIMENT
                LIMIT 20
                """
                context_df = session.sql(context_query).to_pandas()
                
                # Build context
                context_text = "\n".join([
                    f"Product: {row['PRODUCT']}, Review: {row['REVIEW_TEXT'][:150]}, Sentiment: {row['SENTIMENT_SCORE']:.2f}"
                    for _, row in context_df.iterrows()
                ])
                
                # Escape quotes
                context_text = context_text.replace("'", "''")
                qa_question_escaped = qa_question.replace("'", "''")
                
                # Call Cortex Complete
                ai_query = f"""
                SELECT SNOWFLAKE.CORTEX.COMPLETE(
                    'claude-3-5-sonnet',
                    'You are a helpful data analyst. Answer this question about customer reviews:
                    
Question: {qa_question_escaped}

Sample Reviews:
{context_text}

Provide a clear, insightful answer based on the data.'
                ) AS response
                """
                
                response_df = session.sql(ai_query).to_pandas()
                response = response_df['RESPONSE'].iloc[0]
                
                st.markdown("### 🤖 AI Response:")
                st.success(response)
                
            except Exception as e:
                st.error(f"❌ Error generating AI response: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("🚀 Powered by Snowflake Cortex AI | Built with Streamlit")
