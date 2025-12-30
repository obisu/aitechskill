# import packages
import streamlit as st
import pandas as pd
import re
import altair as alt
import json
import os
import snowflake.connector

# Helper function to clean text
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

st.set_page_config(page_title="GenAI Reviews App", page_icon="🤖", layout="wide")

st.title("🤖 Avalanche Streamlit App")
st.write("This is your GenAI-powered data processing app.")

# ============================================================================
# SNOWFLAKE CONNECTION USING ENVIRONMENT VARIABLES
# ============================================================================

@st.cache_resource
def get_snowflake_session():
    """Create Snowflake connection from environment variables"""
    try:
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA"),
        )
        
        # Session wrapper to mimic Snowpark-style .sql().to_pandas()
        class SessionWrapper:
            def __init__(self, conn):
                self.conn = conn
            
            def sql(self, query):
                """Execute SQL and return as pandas DataFrame"""
                cur = self.conn.cursor()
                try:
                    cur.execute(query)
                    rows = cur.fetchall()
                    cols = [c[0] for c in cur.description]
                    df = pd.DataFrame(rows, columns=cols)
                    return df
                finally:
                    cur.close()
            
            def close(self):
                """Close the connection"""
                self.conn.close()
        
        return SessionWrapper(conn)
    
    except Exception as e:
        st.error(f"❌ Failed to connect to Snowflake: {str(e)}")
        st.info("💡 Please check your environment variables")
        return None

# Get session
session = get_snowflake_session()

if session is None:
    st.stop()
else:
    st.sidebar.success("✅ Connected to Snowflake")

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
                query = """
                SELECT * FROM customer_reviews
                LIMIT 1000
                """
                st.session_state["df"] = session.sql(query)
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

    @st.cache_data
    def load_data():
        query_reviews = """
        SELECT *
        FROM REVIEWS_WITH_SENTIMENT
        """
        return session.sql(query_reviews)

    # Load data
    with st.spinner("📥 Loading data..."):
        try:
            df = load_data()
            st.success(f"✅ Loaded {len(df):,} reviews")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.stop()

    # Key Metrics
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

    # Average sentiment by product
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

    # Filter by product
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

    st.subheader(f"📁 Reviews for {product}")
    st.dataframe(filtered_data, use_container_width=True, hide_index=True, height=400)

    # Average sentiment by delivery status
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
    
    # Sentiment Distribution
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
    st.title("🔍 RAG App - Smart Review Search (Avalanche Style)")
    st.markdown("Ask questions about your product reviews")

    prompt = st.text_input(
        "💬 Enter your query:",
        value="Any goggles review?",
        placeholder="e.g., What do customers say about goggles?"
    )

    if prompt:
        if st.button("🔎 Run Query", type="primary"):
            with st.spinner("Searching reviews..."):

                try:
                    safe_prompt = prompt.replace("'", "''")

                    # Simple keyword search (Avalanche style)
                    search_sql = f"""
                        SELECT 
                            REVIEW_TEXT,
                            PRODUCT,
                            SENTIMENT_SCORE
                        FROM REVIEWS_WITH_SENTIMENT
                        WHERE LOWER(REVIEW_TEXT) LIKE LOWER('%{safe_prompt}%')
                        LIMIT 10;
                    """

                    search_df = session.sql(search_sql)

                    if len(search_df) == 0:
                        st.warning("No matching reviews found.")
                    else:
                        st.success(f"Found {len(search_df)} matching reviews")

                        # Show the raw results
                        for idx, row in search_df.iterrows():
                            st.markdown(f"### Result {idx + 1}")
                            st.info(row["REVIEW_TEXT"])
                            st.caption(f"📦 Product: {row['PRODUCT']}")
                            st.caption(f"😊 Sentiment: {row['SENTIMENT_SCORE']:.2f}")
                            st.write("---")

                        # Build context for LLM
                        context = "\n".join(
                            f"- {r['REVIEW_TEXT']}" for _, r in search_df.iterrows()
                        ).replace("'", "''")

                        # Construct the prompt (Avalanche style)
                        full_prompt = f"""
You are a helpful AI assistant. Use the customer review context below to answer the question.

<context>
{context}
</context>

<question>
{safe_prompt}
</question>

Provide a clear, concise answer.
"""

                        # Call Cortex COMPLETE
                        qa_sql = f"""
                            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                                'claude-3-5-sonnet',
                                $$ {full_prompt} $$
                            ) AS ANSWER;
                        """

                        answer_df = session.sql(qa_sql)
                        st.subheader("🤖 AI Summary")
                        st.success(answer_df["ANSWER"].iloc[0])

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")



    # AI-Powered Q&A
    st.markdown("---")
    st.subheader("💬 AI-Powered Q&A")
    
    qa_question = st.text_area(
        "Ask a question about your reviews:",
        placeholder="e.g., What are the main complaints about delivery times?"
    )
    
    if qa_question and st.button("🤖 Get AI Answer", type="secondary"):
        with st.spinner("🤔 Generating answer..."):
            try:
                context_query = """
                SELECT 
                    PRODUCT,
                    REVIEW_TEXT,
                    SENTIMENT_SCORE
                FROM REVIEWS_WITH_SENTIMENT
                LIMIT 20
                """
                context_df = session.sql(context_query)
                
                context_text = "\n".join([
                    f"Product: {row['PRODUCT']}, Review: {row['REVIEW_TEXT'][:150]}, Sentiment: {row['SENTIMENT_SCORE']:.2f}"
                    for _, row in context_df.iterrows()
                ])
                
                context_text = context_text.replace("'", "''")
                qa_question_escaped = qa_question.replace("'", "''")
                
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
                
                response_df = session.sql(ai_query)
                response = response_df['RESPONSE'].iloc[0]
                
                st.markdown("### 🤖 AI Response:")
                st.success(response)
                
            except Exception as e:
                st.error(f"❌ Error generating AI response: {str(e)}")

st.markdown("---")
st.caption("🚀 Powered by Snowflake Cortex AI | Built with Streamlit")
