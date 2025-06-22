# Connecting Clients to Replit-Deployed StreamSwarm Server

## Server URL
Your StreamSwarm server is deployed at:
**https://1c754f36-836a-415b-9318-899ca81361e5-00-zt5cy39lyg1m.janeway.replit.dev**

## Client Connection Commands

### Basic Client Connection
```bash
python client.py --server https://1c754f36-836a-415b-9318-899ca81361e5-00-zt5cy39lyg1m.janeway.replit.dev --name "MyClient-01"
```

### Named Client Examples
```bash
# Office computer
python client.py --server https://1c754f36-836a-415b-9318-899ca81361e5-00-zt5cy39lyg1m.janeway.replit.dev --name "Office-PC"

# Home network
python client.py --server https://1c754f36-836a-415b-9318-899ca81361e5-00-zt5cy39lyg1m.janeway.replit.dev --name "Home-Router"

# Remote location
python client.py --server https://1c754f36-836a-415b-9318-899ca81361e5-00-zt5cy39lyg1m.janeway.replit.dev --name "Remote-Site-A"
```

## Important Notes

1. **Use HTTPS, not HTTP**: Replit deployment requires HTTPS connections
2. **No port number needed**: Replit handles port mapping automatically
3. **Full domain required**: Use the complete Replit domain URL
4. **Client dependencies**: Ensure client machines have `psutil` and `requests` installed

## Troubleshooting

If clients can't connect:
1. Verify the server URL is accessible: Open the URL in a web browser
2. Check client network connectivity: `ping 1c754f36-836a-415b-9318-899ca81361e5-00-zt5cy39lyg1m.janeway.replit.dev`
3. Ensure client has required Python packages: `pip install psutil requests`
4. Check firewall settings on client machines

## Testing Connection
```bash
# Test server accessibility
curl -I https://1c754f36-836a-415b-9318-899ca81361e5-00-zt5cy39lyg1m.janeway.replit.dev

# Should return HTTP/1.1 200 OK
```