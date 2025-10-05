# NFL Betting Suggestions - Web Interface

## 🚀 Quick Start

### 1. Install Web Dependencies
```bash
pip3 install fastapi uvicorn jinja2 httpx
```

### 2. Launch Web Interface
```bash
cd /Users/wilfowler/Sports\ Model/improved_nfl_system
python3 web/launch.py
```

### 3. Access Dashboard
Open: http://localhost:8000

---

## 📋 Pre-Launch Checklist

✅ **Main System Working**
```bash
python3 main.py  # Should generate suggestions
```

✅ **Environment Variables Set**
- ODDS_API_KEY in .env file
- Database exists (created by main.py)

✅ **Dependencies Installed**
```bash
pip3 install -r requirements.txt
```

---

## 🖥️ Web Interface Features

### **Dashboard Overview**
- **Live Suggestions**: Auto-refreshing every hour
- **Three Tiers**: Premium (80+), Standard (65-79), Reference (50-64)
- **Performance Charts**: CLV tracking over 30 days
- **System Status**: API credits, health monitoring

### **Real-Time Updates**
- **HTMX-powered**: Seamless updates without page refresh
- **Smart Polling**: Only when new data available
- **Error Handling**: Graceful degradation on failures

### **Export Capabilities**
- **CSV Export**: Download suggestions as spreadsheet
- **Cache Management**: Clear cache for fresh data
- **Auto-Refresh Toggle**: Enable/disable automatic updates

---

## 🔧 Configuration

### **Environment Variables** (.env)
```bash
# Required
ODDS_API_KEY=your_api_key_here

# Optional Web Settings
WEB_PORT=8000
WEB_HOST=0.0.0.0
SUGGESTIONS_UPDATE_INTERVAL=3600  # 1 hour
PERFORMANCE_UPDATE_INTERVAL=7200  # 2 hours

# Confidence Thresholds
PREMIUM_THRESHOLD=80
STANDARD_THRESHOLD=65
REFERENCE_THRESHOLD=50
```

### **Accessing From Other Devices**
Change WEB_HOST to enable network access:
```bash
export WEB_HOST=0.0.0.0
python3 web/launch.py
```
Then access via: http://YOUR_IP:8000

---

## 🏗️ Architecture

### **Technology Stack**
- **Backend**: FastAPI (fast, type-safe)
- **Frontend**: HTMX + Tailwind CSS (minimal JS)
- **Charts**: Chart.js (performance visualization)
- **Database**: SQLite (your existing database)

### **Data Flow**
```
Main NFL System → Bridge → FastAPI → HTMX → Dashboard
     ↓
  SQLite Database ← CLV Tracking ← Suggestions
```

### **File Structure**
```
web/
├── app.py              # FastAPI application
├── launch.py           # Production launcher
├── bridge/
│   └── nfl_bridge.py   # NFL system integration
├── config/
│   └── web_config.py   # Configuration management
├── templates/
│   ├── base.html       # Base template
│   ├── dashboard.html  # Main dashboard
│   └── error.html      # Error pages
└── tests/
    └── test_integration.py  # Test suite
```

---

## 🧪 Testing

### **Run Tests**
```bash
cd /Users/wilfowler/Sports\ Model/improved_nfl_system
python3 web/tests/test_integration.py
```

### **Manual Testing Checklist**
- [ ] Dashboard loads without errors
- [ ] Suggestions display correctly
- [ ] Charts render with data
- [ ] Auto-refresh works
- [ ] Export functions work
- [ ] Mobile responsive
- [ ] Error handling graceful

---

## 🔍 Troubleshooting

### **Web Interface Won't Start**
```bash
# Check if main system works
python3 main.py

# Check API key
grep ODDS_API_KEY .env

# Check port availability
lsof -i :8000
```

### **No Suggestions Displayed**
1. **Check main system**: Run `python3 main.py` first
2. **Check current week**: System may have no games
3. **Check API credits**: May be exhausted
4. **Check logs**: Look for error messages

### **Charts Not Loading**
1. **Check internet connection**: Chart.js loads from CDN
2. **Check CLV data**: May be empty for new installations
3. **Check browser console**: Look for JavaScript errors

### **Slow Performance**
1. **Reduce update intervals**: Increase SUGGESTIONS_UPDATE_INTERVAL
2. **Check database size**: Large databases may be slow
3. **Check API response times**: The Odds API may be slow

---

## 📊 Performance Metrics

### **Expected Performance**
- **Page Load**: < 2 seconds
- **API Response**: < 1 second
- **Bundle Size**: ~75KB total
- **Memory Usage**: ~50MB RAM

### **Monitoring**
- **System Status**: Available in dashboard sidebar
- **Error Logs**: Check terminal output
- **API Usage**: Tracked in dashboard

---

## 🔄 Updates & Maintenance

### **Daily**
- System automatically updates suggestions
- API credits tracked and displayed
- CLV data collected

### **Weekly**
- Review performance charts
- Check for any error patterns
- Verify API credit usage

### **Monthly**
- Update dependencies if needed
- Review system performance
- Clean old cache data

---

## ⚠️ Important Notes

### **FAIL FAST Philosophy**
- System stops on any critical error
- No fallback data or approximations
- Clear error messages for debugging

### **Data Integrity**
- Only uses REAL data from your main system
- No synthetic or placeholder data
- Respects your API limits

### **Security**
- No external data transmission
- Local network access only
- No user authentication (single user)

---

## 🆘 Support

### **Common Issues**
1. **Port 8000 in use**: Kill existing process or change WEB_PORT
2. **Import errors**: Run `pip3 install -r requirements.txt`
3. **Bridge failures**: Ensure main system works first
4. **Template errors**: Check file permissions

### **Getting Help**
1. Check terminal output for error messages
2. Run tests: `python3 web/tests/test_integration.py`
3. Verify main system: `python3 main.py`
4. Check configuration: Review .env file

---

## 🎯 Success Indicators

✅ **Web interface launches without errors**
✅ **Dashboard displays current suggestions**
✅ **Charts render with actual data**
✅ **Auto-refresh works correctly**
✅ **No JavaScript console errors**
✅ **Mobile interface responsive**
✅ **Export functions work**

Your NFL Betting Suggestions system now has a professional web interface!