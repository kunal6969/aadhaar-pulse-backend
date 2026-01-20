@echo off
cd /d "c:\Users\Kunal\Desktop\UIDAI\aadhaar-pulse-backend"
python -c "import uvicorn; uvicorn.run('app.main:app', host='0.0.0.0', port=8002, reload=False)"
