# 🕐 Aadhaar Pulse Simulator - Backend

A time-traveling analytics platform that serves India's Aadhaar enrollment and update data from **March 1, 2025** to **December 31, 2025**.

## Key Concept

This is NOT a real-time API. All data is historical simulation data. The frontend asks "What did the data look like on June 15, 2025?" and receives data filtered to that simulation date.

## Features

- **Time Simulation**: Query data "as of" any date in the simulation range
- **Enrollment Analytics**: Track Aadhaar enrollment by age group and geography
- **Demographic Updates**: Monitor demographic data updates
- **Biometric Updates**: Track MBU (Mandatory Biometric Update) data
- **ML Forecasting**: Predict future demand using time-series models
- **Anomaly Detection**: Identify unusual patterns in the data
- **Geospatial Heatmaps**: Generate map visualization data

## Tech Stack

- **FastAPI** - Modern, fast web framework
- **SQLAlchemy 2.0** - ORM with SQLite backend
- **Pandas** - Data processing
- **Pydantic** - Request/response validation
- **Uvicorn** - ASGI server

## Project Structure

```
aadhaar-pulse-backend/
├── app/
│   ├── main.py           # FastAPI application entry point
│   ├── config.py         # Configuration settings
│   ├── database.py       # SQLAlchemy setup
│   ├── models/           # SQLAlchemy ORM models
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # Business logic layer
│   ├── routers/          # API endpoints
│   └── utils/            # Utility functions
├── scripts/
│   └── init_database.py  # Database initialization script
├── data/                 # SQLite database and processed files
├── .env                  # Environment configuration
├── requirements.txt      # Python dependencies
├── run.py               # Server startup script
└── README.md
```

## Quick Start

### 1. Clone and Setup

```bash
cd aadhaar-pulse-backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Edit the `.env` file to set the correct paths to your CSV data files:

```env
# CSV Source Directories
ENROLLMENT_CSV_DIR=../api_data_aadhar_enrolment/api_data_aadhar_enrolment
DEMOGRAPHIC_CSV_DIR=../api_data_aadhar_demographic/api_data_aadhar_demographic
BIOMETRIC_CSV_DIR=../api_data_aadhar_biometric/api_data_aadhar_biometric
```

### 3. Initialize Database

Load the CSV data into SQLite:

```bash
python scripts/init_database.py
```

This will:
- Create the SQLite database
- Load all CSV data into the database
- Generate the `available_dates.json` file

### 4. Start the Server

```bash
python run.py
```

The API will be available at:
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Enrollment Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/enrollment/trends` | Get enrollment trends over time |
| GET | `/api/enrollment/summary` | Get aggregated enrollment summary |
| GET | `/api/enrollment/by-district` | Get enrollment by district |
| GET | `/api/enrollment/states` | Get list of all states |
| GET | `/api/enrollment/districts` | Get list of districts for a state |

### Demographic Updates
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/demographic/trends` | Get demographic update trends |
| GET | `/api/demographic/summary` | Get aggregated summary |
| GET | `/api/demographic/by-district` | Get by district breakdown |

### Biometric Updates (MBU)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/biometric/trends` | Get biometric update trends |
| GET | `/api/biometric/summary` | Get aggregated summary |
| GET | `/api/biometric/by-district` | Get by district breakdown |
| GET | `/api/biometric/mbu-risk` | Get MBU risk assessment |

### Analytics & KPIs
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/analytics/kpis` | Get all key performance indicators |
| GET | `/api/analytics/update-burden-index` | Get update burden index by region |
| GET | `/api/analytics/digital-readiness` | Get digital readiness scores |
| GET | `/api/analytics/comparison` | Compare metrics across regions |

### Geospatial
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/geospatial/heatmap/enrollment` | Enrollment intensity heatmap |
| GET | `/api/geospatial/heatmap/demographic` | Demographic update heatmap |
| GET | `/api/geospatial/heatmap/biometric` | Biometric update heatmap |
| GET | `/api/geospatial/heatmap/combined` | Combined activity heatmap |
| GET | `/api/geospatial/centroids/states` | State centroids with lat/lng |
| GET | `/api/geospatial/centroids/districts` | District centroids |

### Forecasting
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/forecast/mbu` | Forecast MBU demand |
| GET | `/api/forecast/trends` | Get trend-based forecast |

### Anomaly Detection
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/anomaly/detect` | Detect anomalies in data |
| GET | `/api/anomaly/recent` | Get recent anomalies |
| GET | `/api/anomaly/summary` | Get anomaly summary |

### Metadata
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/available-dates` | Get all dates with data |
| GET | `/api/simulation-info` | Get simulation metadata |

## Query Parameters

Most endpoints support these common parameters:

- `simulation_date`: The date to query data as of (YYYY-MM-DD format)
- `state`: Filter by state name
- `district`: Filter by district name
- `view_mode`: Aggregation level (daily, weekly, monthly, quarterly)
- `start_date`: Start of date range
- `end_date`: End of date range

## Example Queries

```bash
# Get enrollment trends for a specific date
curl "http://localhost:8000/api/enrollment/trends?simulation_date=2025-06-15"

# Get enrollment summary for a state
curl "http://localhost:8000/api/enrollment/summary?state=Maharashtra"

# Get KPIs for a district
curl "http://localhost:8000/api/analytics/kpis?state=Maharashtra&district=Mumbai"

# Get biometric heatmap data
curl "http://localhost:8000/api/geospatial/heatmap/biometric?simulation_date=2025-09-01"
```

## Data Sources

The API uses three CSV datasets:
- **Enrollment**: New Aadhaar enrollments by age group
- **Demographic Updates**: Demographic data corrections/updates
- **Biometric Updates**: Biometric data refreshes (MBU)

## Development

### Running in development mode

```bash
python run.py
```

The server runs with auto-reload enabled for development.

### Database

The application uses SQLite for simplicity. The database file is created at `data/aadhaar_pulse.db`.

To reset the database:
```bash
rm data/aadhaar_pulse.db
python scripts/init_database.py
```

## License

Internal Project - UIDAI
