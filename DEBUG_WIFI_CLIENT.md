# WiFi Environmental Scanning Debug Guide

## Issue Analysis for Test 191

Based on the investigation, Test 191 (WiFi Environmental Scan) shows no WiFi data because:

1. **Client Setup**: Your client has 2 WiFi interfaces (wlan0 connected, wlp1s0 spare) - this is CORRECT
2. **Missing WiFi Data**: The `wifi_environment_data` field is empty in all test results
3. **Root Cause**: The client likely detected that WiFi scanning is not available

## Test Type Differences

### "WiFi Environmental Scan" Test Type
- **Purpose**: Standalone WiFi environmental analysis
- **Behavior**: 
  - NO ping, bandwidth, or traceroute tests
  - Only WiFi network scanning using spare interface
  - Continuous scans every `interval` seconds for `duration` seconds
  - Results stored in `wifi_environment_data` field only

### "Standard Network Test" Test Type  
- **Purpose**: Regular network performance testing
- **Behavior**:
  - Full ping, bandwidth, traceroute, system metrics
  - PLUS one WiFi environmental scan if spare interface available
  - WiFi data integrated with network performance data

## Client Debug Steps

### 1. Check WiFi Detection on Your Client

Run this on your client machine:
```bash
# Check if iw command is available
iw --version

# Check wireless interfaces
iw dev

# Check interface connection status
iw dev wlan0 link    # Should show connected to SSID
iw dev wlp1s0 link   # Should show "Not connected"

# Test scanning on spare interface
sudo iw dev wlp1s0 scan | head -20
```

### 2. Check Client Logs

When running your client, look for these log messages:
```
INFO - iw command available - WiFi environmental scanning enabled
INFO - Primary WiFi interface detected: wlan0 (connected to YOUR_SSID)
INFO - Spare WiFi interface detected: wlp1s0
INFO - WiFi interface detection complete - Primary: wlan0, Spare: 1
```

### 3. Test WiFi Environmental Scan

If your client starts but WiFi scanning fails, check:
```bash
# Ensure iw is installed
sudo apt-get install iw

# Test scanning permissions
sudo iw dev wlp1s0 scan dump | grep "BSS\|SSID\|signal"
```

### 4. Client Startup Check

Add this to your client startup to verify WiFi detection:
```python
# In client.py, after line 119 (self._detect_wifi_interfaces())
print(f"DEBUG: WiFi interfaces detected: {len(self.wifi_interfaces)}")
print(f"DEBUG: Primary interface: {self.primary_wifi_interface}")
print(f"DEBUG: Spare interfaces: {len(self.spare_wifi_interfaces)} - {self.spare_wifi_interfaces}")
print(f"DEBUG: WiFi scanning available: {WIFI_SCANNING_AVAILABLE}")
```

## Expected Behavior

### For "WiFi Environmental Scan" Test:
- Test results should show NULL values for ping_latency, bandwidth_upload, etc.
- Only `wifi_environment_data` should contain JSON data with:
  - `total_networks`: Number of WiFi networks found
  - `wifi_pollution_score`: Environmental quality score
  - `detected_networks`: Array of network details
  - `channel_usage`: Channel congestion analysis

### For "Standard Network Test" Test:
- Normal network metrics (ping, bandwidth, etc.) PLUS
- `wifi_environment_data` with WiFi environmental analysis
- All metrics integrated in single test result

## Next Steps

1. **Verify your client has WiFi scanning enabled** - check client logs on startup
2. **Run a "Standard Network Test"** to see if integrated WiFi scanning works
3. **Check client permissions** - WiFi scanning may require sudo/root privileges
4. **Verify iw command availability** on your client machine