import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import hashlib
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import os
import time
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import requests

st.set_page_config(page_title="Finance Manager", layout="centered")

# Load environment variables
load_dotenv()

np.random.seed(42)
tf.random.set_seed(42)

def load_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: black;
        font-family: 'Arial', sans-serif;
        background-image: url('https://i.pinimg.com/originals/74/a7/2a/74a72a774e21d0c5904e31e6a6aae9d2.jpg');
    }
    .auth-container {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        max-width: 500px;
        margin: 2rem auto;
    }
    .stButton>button {
        background-color: #4e73df;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2e59d9;
        transform: translateY(-2px);
    }
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4e73df 0%, #224abe 100%);
        color: white;
    }
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stDateInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 1px solid #d1d3e2;
        padding: 8px;
    }
    [data-testid="metric-container"] {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .log-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        overflow-y: auto;
        max-height: 300px;
        font-family: 'Courier New', Courier, monospace;
    }
    </style>
    """, unsafe_allow_html=True)

def create_db_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        return connection
    except Error as e:
        st.error(f"Database connection error: {e}")
        return None

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def create_user(username, password, email):
    conn = create_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, password, email) VALUES (%s, %s, %s)",
                (username, make_hashes(password), email))
            conn.commit()
            st.success("Account created successfully!")
        except Error as e:
            if "Duplicate entry" in str(e):
                st.error("Username already exists")
            else:
                st.error(f"Error creating user: {e}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

def login_user(username, password):
    conn = create_db_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            if user and check_hashes(password, user['password']):
                return True
            return False
        except Error as e:
            st.error(f"Login error: {e}")
            return False
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def add_transaction(username, trans_type, category, amount, date, description):
    conn = create_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO transactions 
                (username, type, category, amount, date, description) 
                VALUES (%s, %s, %s, %s, %s, %s)""",
                (username, trans_type, category, amount, date, description))
            conn.commit()
            st.success("Transaction added successfully!")
        except Error as e:
            st.error(f"Error adding transaction: {e}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

def get_transactions(username):
    conn = create_db_connection()
    if conn:
        try:
            query = f"""
                SELECT 
                    id, type, category, amount, 
                    DATE_FORMAT(date, '%Y-%m-%d') as date, 
                    description 
                FROM transactions 
                WHERE username = '{username}'
                ORDER BY date DESC
            """
            df = pd.read_sql(query, conn)
            return df
        except Error as e:
            st.error(f"Error fetching transactions: {e}")
            return pd.DataFrame()
        finally:
            if conn.is_connected():
                conn.close()
def fetch_stock_data(ticker, start_date=None, end_date=None, max_retries=3):
    """Fetch historical stock data from Yahoo Finance with retry logic"""
    if start_date is None:
        start_date = "2020-01-01"
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    session = requests.Session()
    session.headers = headers
    
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, session=session)
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            return data['Close']  # Use Close price instead of Adj Close for consistency
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}/{max_retries} for {ticker}: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to download {ticker} after {max_retries} attempts: {str(e)}")
                return None

def prepare_data(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        Dense(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return predictions, rmse

def analyze_stock(ticker, start_date, end_date, seq_length=30, epochs=50, daystopredict=30):
    logs = []
    
    def log(message):
        logs.append(message)
        print(message)

    try:
        log(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        prices = fetch_stock_data(ticker, start_date, end_date)
        
        if prices is None:
            log("Failed to fetch data")
            return None, logs
            
        log(f"Data fetched: {len(prices)} days")

        if len(prices) < seq_length * 2:
            min_required_days = seq_length * 2
            extended_start_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
            log(f"Insufficient data ({len(prices)} days). Trying extended range from {extended_start_date}...")
            prices = fetch_stock_data(ticker, extended_start_date, end_date)
            if prices is None or len(prices) < seq_length * 2:
                log(f"Not enough data for {ticker} (need {min_required_days} days, got {len(prices) if prices is not None else 0})")
                return None, logs

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices.values.reshape(-1, 1))
        log(f"Scaled data shape: {scaled_data.shape}")

        X, y = prepare_data(scaled_data, seq_length)
        log(f"Prepared data: X shape {X.shape}, y shape {y.shape}")

        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        log(f"Train/Test split: {len(X_train)}/{len(X_test)} samples")

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        log(f"Reshaped X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        log(f"Building and training LSTM model for {ticker}...")
        model = build_lstm_model((seq_length, 1))
        model.fit(X_train, y_train,
                  batch_size=32,
                  epochs=epochs,
                  validation_data=(X_test, y_test),
                  verbose=0)
        log("Model training completed")

        predictions, rmse = evaluate_model(model, X_test, y_test, scaler)
        log(f"Evaluation completed, RMSE: {rmse}")

        test_dates = prices.index[split + seq_length:]
        last_n_days = scaled_data[-seq_length:]
        last_n_days = last_n_days.reshape(1, seq_length, 1)

        all_predictions = []
        for _ in range(daystopredict):
            next_day_pred = model.predict(last_n_days)
            next_day_price = scaler.inverse_transform(next_day_pred)[0][0]
            all_predictions.append(next_day_price)
            
            last_n_days = np.roll(last_n_days, shift=-1, axis=1)
            last_n_days[0, -1, 0] = next_day_pred[0][0]

        log(f"Predicted prices for the next {daystopredict} days: {all_predictions}")

        # Get current price with custom headers
        session = requests.Session()
        session.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        current_data = yf.download(ticker, period='1d', progress=False, session=session)
        if current_data.empty:
            raise ValueError("No current data available")
        current_price = current_data['Close'].iloc[-1]
        log(f"Current price: {current_price}")

        potential_gain = (all_predictions[0] - current_price) / current_price * 100

        result = {
            'ticker': ticker,
            'model': model,
            'rmse': rmse,
            'predicted_prices': all_predictions,
            'current_price': current_price,
            'potential_gain': potential_gain,
            'prices': prices,
            'test_dates': test_dates,
            'predictions': predictions
        }
        return result, logs

    except Exception as e:
        log(f"Error analyzing {ticker}: {str(e)}")
        return None, logs

def plot_results(result):
    if result is None:
        return

    plt.figure(figsize=(15, 5))
    plt.plot(result['prices'].index, result['prices'], label='Historical', color='blue', alpha=0.5)
    plt.plot(result['test_dates'], result['prices'][-len(result['test_dates']):],
             label='Actual (Test)', color='blue')
    plt.plot(result['test_dates'], result['predictions'],
             label='Predicted', color='red', linestyle='--')

    last_date = result['prices'].index[-1]
    current_price = result['current_price'].iloc[0] if isinstance(result['current_price'], pd.Series) else result['current_price']
    plt.plot(last_date, current_price, 'go', label=f'Current: ${current_price:.2f}')

    predicted_dates = [last_date + timedelta(days=i) for i in range(1, len(result['predicted_prices']) + 1)]
    for i, pred_price in enumerate(result['predicted_prices']):
        plt.plot(predicted_dates[i], pred_price, 'ro', label=f'Predicted Day {i+1}: ${pred_price:.2f}')

    rmse_value = result['rmse'].iloc[0] if isinstance(result['rmse'], pd.Series) else result['rmse']
    potential_gain_value = result["potential_gain"].iloc[0] if isinstance(result["potential_gain"], pd.Series) else result["potential_gain"]

    plt.title(f'{result["ticker"]} Stock Price Prediction\nRMSE: {rmse_value:.2f} | Potential Gain: {potential_gain_value:.2f}%')

    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

def predict_stock_sklearn(ticker, seq_length=30):
    logs = []

    def log(message):
        logs.append(message)
        print(message)

    prices = fetch_stock_data(ticker)
    if prices is None or len(prices) < seq_length * 2:
        log(f"Not enough data for {ticker}")
        return None, logs, None

    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices.values.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(len(scaled_prices) - seq_length):
        X.append(scaled_prices[i:i+seq_length])
        y.append(scaled_prices[i+seq_length])

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    last_seq = scaled_prices[-seq_length:]
    next_pred_scaled = model.predict([last_seq])[0]
    next_pred = scaler.inverse_transform([[next_pred_scaled]])[0][0]

    return next_pred, logs, prices

def login_page():
    st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
    st.title("💰 Finance Manager Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        if submit_button:
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")
    if st.button("Create New Account"):
        st.session_state.show_signup = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def signup_page():
    st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
    st.title("📝 Create Account")
    with st.form("signup_form"):
        new_username = st.text_input("Username")
        new_email = st.text_input("Email")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit_button = st.form_submit_button("Sign Up")
        if submit_button:
            if new_password == confirm_password:
                create_user(new_username, new_password, new_email)
            else:
                st.error("Passwords do not match")
    if st.button("Back to Login"):
        st.session_state.show_signup = False
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def stock_prediction_page():
    st.title("📈 Stock Market Predictor")
    
    menu = ["LSTM Prediction", "Random Forest Prediction", "Stock Suggestions"]
    choice = st.sidebar.selectbox("Prediction Method", menu)
    
    if choice == "LSTM Prediction":
        st.markdown("<div class='stock-prediction-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Stock Symbol (e.g., AAPL)", "AAPL").upper()
        with col2:
            days_to_predict = st.number_input("Days to Predict", 1, 365, 30)
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2 * 365)).strftime('%Y-%m-%d')
        
        if st.button("Generate LSTM Prediction"):
            with st.spinner("Analyzing stock data with LSTM..."):
                result, logs = analyze_stock(ticker, start_date, end_date, epochs=30, daystopredict=days_to_predict)
                
                if result is not None:
                    st.session_state.stock_result = result
                    plot_results(result)
                    
                    if isinstance(result['predicted_prices'], pd.Series):
                        predicted_prices = result['predicted_prices'].to_list()  # Convert Series to list

                    else:
                        predicted_prices = result['predicted_prices']  

                        current_price = result['current_price'].iloc[0] if isinstance(result['current_price'], pd.Series) else result['current_price']
                        potential_gain = result['potential_gain'].iloc[0] if isinstance(result['potential_gain'], pd.Series) else result['potential_gain']
                        rmse = result['rmse'].iloc[0] if isinstance(result['rmse'], pd.Series) else result['rmse']
                        predicted_prices = list(predicted_prices) if isinstance(predicted_prices, pd.Series) else predicted_prices

                        st.markdown(f"### 📊 Trading Recommendations")
                        st.markdown(f"""
                        - **Current Price**: ${current_price:.2f}
                        - **Predicted Prices for the Next {days_to_predict} Days**: 
                        {', '.join([f'${price:.2f}' for price in predicted_prices])}
                        - **Potential Gain**: {potential_gain:.2f}%
                        - **RMSE**: {rmse:.2f}
                        """)
                    
                    st.markdown("<div class='log-container'>", unsafe_allow_html=True)
                    st.subheader("Analysis Logs")
                    st.text("\n".join(logs))
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Failed to analyze {ticker}.")
                    st.markdown("<div class='log-container'>", unsafe_allow_html=True)
                    st.subheader("Analysis Logs")
                    st.text("\n".join(logs))
                    st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif choice == "Random Forest Prediction":
        st.markdown("<div class='stock-prediction-card'>", unsafe_allow_html=True)
        st.subheader("Random Forest Stock Prediction")
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):")

        if ticker:
            prediction, logs, prices = predict_stock_sklearn(ticker)
            if prediction:
                current_price = prices.iloc[-1].item()

                st.success(f"Predicted Next Day Closing Price: ${prediction:.2f}")
                st.info(f"Current Closing Price: ${current_price:.2f}")

                st.subheader("📊 Price Comparison")
                price_df = pd.DataFrame({
                    'Type': ['Current Price', 'Predicted Price'],
                    'Price': [current_price, prediction]
                })
                st.bar_chart(price_df.set_index('Type'))

                st.subheader("📈 Past Trend + Prediction")
                n_days = st.slider("Select number of past days to display", min_value=30, max_value=180, value=60)
                past_prices = prices[-n_days:].copy()
                future_date = pd.Timestamp.today() + pd.Timedelta(days=1)
                trend_df = pd.concat([past_prices, pd.Series([prediction], index=[future_date])])
                trend_df.name = 'Price'
                st.line_chart(trend_df)

            with st.expander("Show Logs"):
                for log in logs:
                    st.text(log)
        st.markdown("</div>", unsafe_allow_html=True)
    
    elif choice == "Stock Suggestions":
        st.markdown("<div class='stock-prediction-card'>", unsafe_allow_html=True)
        st.subheader("💡 Investment Suggestions")

        st.markdown("Here are a few trending ideas based on market sentiment:")

        with st.expander("🔍 High Growth Tech Stocks"):
            st.info("""
            - NVDA (NVIDIA): Leading AI chipmaker.
            - AMD: Competitor in the graphics and CPU sector.
            - AAPL: Stable long-term growth with strong fundamentals.
            """)

        with st.expander("📦 Consumer Essentials"):
            st.info("""
            - PG (Procter & Gamble): Strong dividend payout.
            - KO (Coca-Cola): Resilient to economic downturns.
            - COST (Costco): Retail with stable revenue streams.
            """)

        with st.expander("🛢 Energy & Commodities"):
            st.info("""
            - XOM (Exxon Mobil): Rising oil prices.
            - BP: Growth in clean energy + traditional oil.
            """)

        st.warning("📌 Disclaimer: These are not financial recommendations. Please do your own research before investing.")
        st.markdown("</div>", unsafe_allow_html=True)

def delete_transaction(transaction_id):
    conn = create_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transactions WHERE id = %s", (transaction_id,))
            conn.commit()
            st.success("Transaction deleted successfully!")
            return True
        except Error as e:
            st.error(f"Error deleting transaction: {e}")
            return False
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

def dashboard_page():
    st.sidebar.title(f"Welcome, {st.session_state.username}")
    menu = ["Dashboard", "Add Transaction", "Delete Transaction","Transaction History", "Stock Predictor"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Dashboard":
        show_dashboard()
    elif choice == "Add Transaction":
        add_transaction_page()
    elif choice=="Delete Transaction":
        del_transaction()
    elif choice == "Transaction History":
        transaction_history_page()
    elif choice == "Stock Predictor":
        stock_prediction_page()
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

def show_dashboard():
    st.title("📊 Financial Dashboard")
    goal = st.text_input("Enter your goal:")
    if goal:
        st.write(f"Your goal is: {goal}")
    df = get_transactions(st.session_state.username)
    if not df.empty:
        total_income = df[df['type'] == 'income']['amount'].sum()
        total_expense = df[df['type'] == 'expense']['amount'].sum()
        balance = total_income - total_expense
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Income", f"${total_income:,.2f}")
        with col2:
            st.metric("Total Expenses", f"${total_expense:,.2f}")
        with col3:
            st.metric("Current Balance", f"${balance:,.2f}")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("💰 Expense Breakdown")
        if not df[df['type'] == 'expense'].empty:
            expense_df = df[df['type'] == 'expense']
            fig = px.pie(expense_df, values='amount', names='category', title='Expense Categories')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expense data available")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("💵 Income Breakdown")
        if not df[df['type'] == 'income'].empty:
            income_df = df[df['type'] == 'income']
            fig = px.pie(income_df, values='amount', names='category', title='Income Sources')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No income data available")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No transactions found. Add your first transaction!")

def add_transaction_page():
    st.title("➕ Add Transaction")
    with st.form("transaction_form"):
        trans_type = st.selectbox("Type", ["income", "expense"])
        category = st.text_input("Category")
        amount = st.number_input("Amount", min_value=0.01, step=0.01)
        date = st.date_input("Date", datetime.now())
        description = st.text_area("Description")
        submitted = st.form_submit_button("Add Transaction")
        if submitted:
            add_transaction(st.session_state.username, trans_type, category, amount, date, description)

def transaction_history_page():
    st.title("📜 Transaction History")
    df = get_transactions(st.session_state.username)
    if not df.empty:
        col1, col2 = st.columns(2)
        with col1:
            filter_type = st.selectbox("Filter by type", ["All"] + list(df['type'].unique()))
        with col2:
            filter_category = st.selectbox("Filter by category", ["All"] + list(df['category'].unique()))
        if filter_type != "All":
            df = df[df['type'] == filter_type]
        if filter_category != "All":
            df = df[df['category'] == filter_category]
        columns_to_drop = ['username'] if 'username' in df.columns else []
        st.dataframe(df.drop(columns=columns_to_drop).style.format({'amount': '${:,.2f}'}), use_container_width=True)
        st.download_button(label="Download as CSV", data=df.to_csv(index=False), file_name='transactions.csv', mime='text/csv')
    else:
        st.info("No transactions found")

def main():
    load_css()
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.show_signup = False
        st.session_state.username = None
    
    if not st.session_state.logged_in:
        if st.session_state.show_signup:
            signup_page()
        else:
            login_page()
    else:
        dashboard_page()

if __name__ == "__main__":
    main()
