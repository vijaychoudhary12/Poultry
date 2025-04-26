import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import logging
import warnings
import holidays
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_absolute_error
from functools import lru_cache
import json
import pytz

warnings.filterwarnings("ignore")

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'cache_timeout': 3600,  # 1 hour cache
    'max_train_years': 10,
    'min_data_points': 30,
    'holiday_country': 'IN',
    'timezone': 'Asia/Kolkata',
    'default_db': 'necc_prices.db',
    'district_db': 'nearest_necc.db'
}

# Helper functions
def get_db_connection(db_name=None):
    """Get a connection to the NECC prices database"""
    db = db_name if db_name else CONFIG['default_db']
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn

def get_ist_time():
    return datetime.now(pytz.timezone(CONFIG['timezone']))

def calculate_holidays(year):
    india_holidays = holidays.CountryHoliday(CONFIG['holiday_country'], years=year)
    return list(india_holidays.keys())

def validate_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return None

# Enhanced prediction models
class PricePredictor:
    def __init__(self, city):
        self.city = city
        self.current_date = get_ist_time().date()
        
    def load_data(self, start_date=None, end_date=None):
        """Load price data for the city with optional date range"""
        if start_date is None:
            start_date = (self.current_date - timedelta(days=365*CONFIG['max_train_years'])).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = self.current_date.strftime('%Y-%m-%d')
            
        conn = get_db_connection()
        query = '''
            SELECT Date, Price 
            FROM Daily_Prices 
            WHERE City = ? AND Date BETWEEN ? AND ? AND Price IS NOT NULL
            ORDER BY Date
        '''
        data = conn.execute(query, (self.city, start_date, end_date)).fetchall()
        conn.close()
        
        if not data or len(data) < CONFIG['min_data_points']:
            logger.warning(f"Insufficient data for {self.city}: {len(data)} records")
            return None
            
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        return df['Price']
    
    def holt_winters_predict(self, data, periods=12):
        """Holt-Winters Exponential Smoothing prediction"""
        try:
            model = ExponentialSmoothing(
                data,
                trend='add',
                seasonal='mul',
                seasonal_periods=12,
                initialization_method='estimated'
            ).fit()
            forecast = model.forecast(periods)
            return forecast
        except Exception as e:
            logger.error(f"Holt-Winters failed for {self.city}: {str(e)}")
            return None
    
    def arima_predict(self, data, periods=12):
        """ARIMA prediction"""
        try:
            model = ARIMA(data, order=(1,1,1)).fit()
            forecast = model.forecast(steps=periods)
            return forecast
        except Exception as e:
            logger.error(f"ARIMA failed for {self.city}: {str(e)}")
            return None
    
    def prophet_predict(self, data, periods=12):
        """Facebook Prophet prediction"""
        try:
            df = data.reset_index()
            df.columns = ['ds', 'y']
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            model.fit(df)
            
            future = model.make_future_dataframe(periods=periods, freq='M')
            forecast = model.predict(future)
            
            return forecast.set_index('ds')['yhat'][-periods:]
        except Exception as e:
            logger.error(f"Prophet failed for {self.city}: {str(e)}")
            return None
    
    def ensemble_predict(self, data, periods=12):
        """Ensemble prediction using multiple models"""
        predictions = []
        models = {
            'holt_winters': self.holt_winters_predict(data, periods),
            'arima': self.arima_predict(data, periods),
            'prophet': self.prophet_predict(data, periods)
        }
        
        # Only use successful predictions
        valid_preds = {k: v for k, v in models.items() if v is not None}
        
        if not valid_preds:
            return None
            
        # Calculate ensemble as simple average
        ensemble = pd.concat(valid_preds.values(), axis=1).mean(axis=1)
        return ensemble
    
    def predict(self, prediction_type='next_12months'):
        """Main prediction method with multiple forecast types"""
        # Always work with monthly data for these predictions
        data = self.load_data()
        if data is None:
            return None
            
        monthly_data = data.resample('M').mean()
        
        if prediction_type == 'next_month':
            periods = 1
        elif prediction_type == 'next_3months':
            periods = 3
        elif prediction_type == 'next_6months':
            periods = 6
        elif prediction_type == 'next_year':
            periods = 12
        else:  # next_12months
            periods = 12
            
        # Get predictions from all models
        pred_hw = self.holt_winters_predict(monthly_data, periods)
        pred_arima = self.arima_predict(monthly_data, periods)
        pred_prophet = self.prophet_predict(monthly_data, periods)
        pred_ensemble = self.ensemble_predict(monthly_data, periods)
        
        # Determine which predictions to return
        pred = pred_ensemble if pred_ensemble is not None else pred_hw
        
        if pred is None:
            return None
            
        # Create dates for the forecast period
        if prediction_type == 'next_month':
            dates = [self.current_date.replace(day=1) + timedelta(days=32)]
            dates = [d.replace(day=1) - timedelta(days=1) for d in dates]  # Last day of next month
        else:
            dates = pd.date_range(
                start=self.current_date.replace(day=1), 
                periods=periods, 
                freq='M'
            )
            
        result = {
            'prediction': pred.tolist(),
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'models': list(valid_preds.keys()),
            'last_actual': monthly_data.iloc[-1] if len(monthly_data) > 0 else None,
            'last_date': monthly_data.index[-1].strftime('%Y-%m-%d') if len(monthly_data) > 0 else None,
            'holt_winters': pred_hw.tolist() if pred_hw is not None else None,
            'arima': pred_arima.tolist() if pred_arima is not None else None,
            'prophet': pred_prophet.tolist() if pred_prophet is not None else None,
            'ensemble': pred_ensemble.tolist() if pred_ensemble is not None else None
        }
        
        return result

# API Endpoints
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    conn = get_db_connection()
    try:
        cities = conn.execute('SELECT DISTINCT City FROM Daily_Prices ORDER BY City').fetchall()
    except sqlite3.OperationalError as e:
        logger.error(f"Database error: {str(e)}")
        return render_template('error.html', message="Database configuration error"), 500
    finally:
        conn.close()
    
    district_conn = get_db_connection(CONFIG['district_db'])
    try:
        districts = district_conn.execute('SELECT district FROM district_mappings ORDER BY district').fetchall()
    except sqlite3.OperationalError as e:
        logger.error(f"District database error: {str(e)}")
        districts = []
    finally:
        district_conn.close()
    
    return render_template('dashboard.html', 
                         cities=[city['City'] for city in cities],
                         districts=[dist['district'] for dist in districts],
                         datetime=get_ist_time())

@app.route('/api/cities')
def get_cities():
    conn = get_db_connection()
    cities = conn.execute('SELECT DISTINCT City FROM Daily_Prices ORDER BY City').fetchall()
    conn.close()
    return jsonify({'cities': [city['City'] for city in cities]})

@app.route('/api/districts')
def get_districts():
    conn = get_db_connection(CONFIG['district_db'])
    districts = conn.execute('SELECT district FROM district_mappings ORDER BY district').fetchall()
    conn.close()
    return jsonify({'districts': [d['district'] for d in districts]})

@app.route('/api/price')
def get_daily_price():
    city = request.args.get('city')
    date = request.args.get('date', get_ist_time().strftime('%Y-%m-%d'))
    
    if not city:
        return jsonify({'error': 'City parameter is required'}), 400
    
    if not validate_date(date):
        return jsonify({'error': 'Invalid date format (use YYYY-MM-DD)'}), 400
    
    conn = get_db_connection()
    if city.lower() == 'all':
        prices = conn.execute('''
            SELECT City, Price 
            FROM Daily_Prices 
            WHERE Date = ? AND Price IS NOT NULL
            ORDER BY Price DESC
        ''', (date,)).fetchall()
        conn.close()
        
        if prices:
            return jsonify({
                'date': date,
                'prices': [{'city': row['City'], 'price': row['Price']} for row in prices]
            })
        return jsonify({'error': 'No data available'}), 404
    else:
        price = conn.execute('''
            SELECT Price 
            FROM Daily_Prices 
            WHERE City = ? AND Date = ?
        ''', (city, date)).fetchone()
        conn.close()
        
        if price and price['Price'] is not None:
            return jsonify({
                'date': date,
                'city': city,
                'price': price['Price']
            })
        return jsonify({'error': 'Data unavailable'}), 404

@app.route('/api/price/range')
def get_price_range():
    city = request.args.get('city')
    start_date = request.args.get('start', (get_ist_time() - timedelta(days=7)).strftime('%Y-%m-%d'))
    end_date = request.args.get('end', get_ist_time().strftime('%Y-%m-%d'))
    
    if not city:
        return jsonify({'error': 'City parameter is required'}), 400
    
    if not validate_date(start_date) or not validate_date(end_date):
        return jsonify({'error': 'Invalid date format (use YYYY-MM-DD)'}), 400
    
    conn = get_db_connection()
    if city.lower() == 'all':
        prices = conn.execute('''
            SELECT City, Date, Price 
            FROM Daily_Prices 
            WHERE Date BETWEEN ? AND ? AND Price IS NOT NULL
            ORDER BY Date, City
        ''', (start_date, end_date)).fetchall()
        conn.close()
        
        if prices:
            return jsonify({
                'start_date': start_date,
                'end_date': end_date,
                'prices': [{
                    'city': row['City'],
                    'date': row['Date'],
                    'price': row['Price']
                } for row in prices]
            })
        return jsonify({'error': 'No data available'}), 404
    else:
        prices = conn.execute('''
            SELECT Date, Price 
            FROM Daily_Prices 
            WHERE City = ? AND Date BETWEEN ? AND ? AND Price IS NOT NULL
            ORDER BY Date
        ''', (city, start_date, end_date)).fetchall()
        conn.close()
        
        if prices:
            return jsonify({
                'city': city,
                'start_date': start_date,
                'end_date': end_date,
                'prices': [{
                    'date': row['Date'],
                    'price': row['Price']
                } for row in prices]
            })
        return jsonify({'error': 'No data available'}), 404

@app.route('/api/monthly')
def get_monthly_avg():
    city = request.args.get('city')
    year = request.args.get('year')
    month = request.args.get('month')
    
    if not all([city, year, month]):
        return jsonify({'error': 'City, year and month parameters are required'}), 400
    
    try:
        year = int(year)
        month = int(month)
    except ValueError:
        return jsonify({'error': 'Year and month must be integers'}), 400
    
    conn = get_db_connection()
    prices = conn.execute('''
        SELECT Date, Price 
        FROM Daily_Prices 
        WHERE City = ? AND strftime('%Y', Date) = ? AND strftime('%m', Date) = ? AND Price IS NOT NULL
    ''', (city, f"{year:04d}", f"{month:02d}")).fetchall()
    conn.close()
    
    if prices:
        avg_price = sum(row['Price'] for row in prices) / len(prices)
        return jsonify({
            'city': city,
            'year': year,
            'month': month,
            'avg_price': avg_price,
            'data_points': len(prices)
        })
    return jsonify({'error': 'No data available'}), 404

@app.route('/api/monthly/range')
def get_monthly_range():
    city = request.args.get('city')
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    if not all([city, start_date, end_date]):
        return jsonify({'error': 'City, start and end parameters are required'}), 400
    
    if not validate_date(start_date) or not validate_date(end_date):
        return jsonify({'error': 'Invalid date format (use YYYY-MM-DD)'}), 400
    
    conn = get_db_connection()
    prices = conn.execute('''
        SELECT strftime('%Y', Date) as year, strftime('%m', Date) as month, AVG(Price) as avg_price
        FROM Daily_Prices 
        WHERE City = ? AND Date BETWEEN ? AND ? AND Price IS NOT NULL
        GROUP BY year, month
        ORDER BY year, month
    ''', (city, start_date, end_date)).fetchall()
    conn.close()
    
    if prices:
        return jsonify({
            'city': city,
            'start_date': start_date,
            'end_date': end_date,
            'avgs': [{
                'year': row['year'],
                'month': row['month'],
                'avg_price': row['avg_price']
            } for row in prices]
        })
    return jsonify({'error': 'No data available'}), 404

@app.route('/api/yearly')
def get_yearly_avg():
    city = request.args.get('city')
    year = request.args.get('year')
    
    if not all([city, year]):
        return jsonify({'error': 'City and year parameters are required'}), 400
    
    try:
        year = int(year)
    except ValueError:
        return jsonify({'error': 'Year must be an integer'}), 400
    
    conn = get_db_connection()
    prices = conn.execute('''
        SELECT AVG(Price) as avg_price
        FROM Daily_Prices 
        WHERE City = ? AND strftime('%Y', Date) = ? AND Price IS NOT NULL
    ''', (city, f"{year:04d}")).fetchone()
    conn.close()
    
    if prices and prices['avg_price'] is not None:
        return jsonify({
            'city': city,
            'year': year,
            'avg_price': prices['avg_price']
        })
    return jsonify({'error': 'No data available'}), 404

@app.route('/api/yearly/range')
def get_yearly_range():
    city = request.args.get('city')
    start_year = request.args.get('start')
    end_year = request.args.get('end')
    
    if not all([city, start_year, end_year]):
        return jsonify({'error': 'City, start and end parameters are required'}), 400
    
    try:
        start_year = int(start_year)
        end_year = int(end_year)
    except ValueError:
        return jsonify({'error': 'Start and end must be integers'}), 400
    
    conn = get_db_connection()
    prices = conn.execute('''
        SELECT strftime('%Y', Date) as year, AVG(Price) as avg_price
        FROM Daily_Prices 
        WHERE City = ? AND strftime('%Y', Date) BETWEEN ? AND ? AND Price IS NOT NULL
        GROUP BY year
        ORDER BY year
    ''', (city, f"{start_year:04d}", f"{end_year:04d}")).fetchall()
    conn.close()
    
    if prices:
        return jsonify({
            'city': city,
            'start_year': start_year,
            'end_year': end_year,
            'avgs': [{
                'year': row['year'],
                'avg_price': row['avg_price']
            } for row in prices]
        })
    return jsonify({'error': 'No data available'}), 404

@app.route('/api/predict')
@lru_cache(maxsize=100)
def get_prediction():
    city = request.args.get('city')
    prediction_type = request.args.get('type', 'next_12months')
    
    if not city:
        return jsonify({'error': 'City parameter is required'}), 400
    
    # Only allow these prediction types
    if prediction_type not in ['next_month', 'next_3months', 'next_6months', 'next_year', 'next_12months']:
        return jsonify({'error': 'Invalid prediction type'}), 400
    
    predictor = PricePredictor(city)
    result = predictor.predict(prediction_type)
    
    if result is None:
        return jsonify({'error': 'Prediction failed'}), 500
        
    return jsonify({
        'city': city,
        'type': prediction_type,
        'prediction': result['prediction'],
        'dates': result['dates'],
        'models': result['models'],
        'last_actual': result['last_actual'],
        'last_date': result['last_date'],
        'holt_winters': result['holt_winters'],
        'arima': result['arima'],
        'prophet': result['prophet'],
        'ensemble': result['ensemble']
    })

@app.route('/api/district/price')
def get_district_price():
    district = request.args.get('district')
    
    if not district:
        return jsonify({'error': 'District parameter is required'}), 400
    
    conn = get_db_connection(CONFIG['district_db'])
    mapping = conn.execute('''
        SELECT nearest_necc_city, distance 
        FROM district_mappings 
        WHERE district = ?
    ''', (district,)).fetchone()
    conn.close()
    
    if not mapping:
        return jsonify({'error': 'District not found'}), 404
        
    nearest_city = mapping['nearest_necc_city']
    distance = mapping['distance']
    
    if nearest_city == 'N/A':
        return jsonify({
            'district': district,
            'price': None,
            'based_on': None,
            'distance': None,
            'error': 'No nearby NECC city'
        })
    
    try:
        # Get current price for the nearest city
        conn = get_db_connection()
        price = conn.execute('''
            SELECT Price 
            FROM Daily_Prices 
            WHERE City = ? AND Date <= ?
            ORDER BY Date DESC 
            LIMIT 1
        ''', (nearest_city, datetime.now().strftime('%Y-%m-%d'))).fetchone()
        conn.close()
        
        if not price or price['Price'] is None:
            return jsonify({
                'district': district,
                'price': None,
                'based_on': nearest_city,
                'distance': distance,
                'error': 'Price data unavailable'
            })
        
        # Get predictions safely
        predictions = {}
        try:
            predictor = PricePredictor(nearest_city)
            
            for pred_type in ['next_month', 'next_3months', 'next_year', 'next_12months']:
                try:
                    result = predictor.predict(pred_type)
                    if result:
                        predictions[pred_type] = {
                            'prediction': result['prediction'][0] if isinstance(result['prediction'], list) else result['prediction'],
                            'dates': result['dates'],
                            'models': result['models']
                        }
                except Exception as e:
                    logger.error(f"Error predicting {pred_type} for {nearest_city}: {str(e)}")
                    predictions[pred_type] = None
        except Exception as e:
            logger.error(f"Error creating predictor for {nearest_city}: {str(e)}")
            predictions = {
                'next_month': None,
                'next_3months': None,
                'next_year': None,
                'next_12months': None
            }
        
        return jsonify({
            'district': district,
            'price': price['Price'],
            'based_on': nearest_city,
            'distance': distance,
            'predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"Error getting district price for {district}: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)