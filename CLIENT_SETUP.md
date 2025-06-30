# StreamSwarm Client Setup Guide

## Basic Client Setup

The basic client functionality works with just:
1. Copy `client.py` to your client machine
2. Run: `python client.py --server-url http://your-server:5000 --api-token your-token`

## Enhanced Features (Optional)

### Geolocation Path Analysis (Interactive Maps)

To enable the interactive world map feature that shows the geographic path of network traffic:

**Required Files:**
- Copy `geolocation_service.py` to the same directory as `client.py`

**Required Python Packages:**
```bash
pip install folium requests
```

**System Requirements:**
- `traceroute` command available (usually pre-installed on Linux/macOS)
- Internet connection for geolocation API lookups

### Advanced Network Testing

For full network analysis capabilities:

```bash
pip install speedtest-cli scapy psutil
```

**System Requirements:**
- Root/admin privileges may be needed for some advanced network tests
- Network tools: `ping`, `traceroute`, `mtr` (optional)

## Troubleshooting

If geolocation maps don't appear:
1. Check that `geolocation_service.py` is in the same directory as `client.py`
2. Verify `folium` package is installed: `python -c "import folium"`
3. Check client logs for import errors
4. Ensure client has internet access for geolocation lookups

## Alternative: Server-Side Processing

If you prefer not to install additional packages on clients, the system can process traceroute data on the server side. Contact your administrator to enable server-side geolocation processing.