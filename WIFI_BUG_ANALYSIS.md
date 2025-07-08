# WiFi Environmental Scan Bug Analysis

## Issue Summary
Client receives `test_type: 'standard'` instead of `'wifi_environment'` even when WiFi Environmental Scan tests are created correctly.

## Evidence Collected

### 1. Client Hardware Status ✓
- **Two WiFi interfaces**: wlan0 (connected), wlp1s0 (spare)
- **iw command available**: Version 6.9 working properly
- **WiFi scanning functional**: Successfully scans networks
- **Client detection working**: Properly classifies primary vs spare interfaces

### 2. Database Storage ✓
- **Test 191**: `test_type: wifi_environment` ✓
- **Test 192**: `test_type: wifi_environment` ✓  
- **Test 193**: `test_type: wifi_environment` ✓
- All tests correctly stored in database with WiFi test type

### 3. Client Debug Output ❌
```
2025-07-08 06:25:59,256 - INFO - Starting test 193 to vimeo.com for 400s (type: standard)
2025-07-08 06:25:59,256 - INFO - DEBUG: Test type received: 'standard' (type: <class 'str'>)
```

### 4. Server API Analysis
- **Route**: `/api/client/<int:client_id>/tests`
- **Function**: `get_client_tests()` calls `test.to_dict()`
- **Model**: `Test.to_dict()` includes `'test_type': self.test_type`
- **Added Debug**: Server-side logging to trace test serialization

## Root Cause Hypothesis

The issue is in the **server-to-client communication layer**. The database stores the correct test_type, but something in the API response is overriding it to 'standard'.

## Possible Causes

1. **API Response Override**: Something in the response processing changes test_type
2. **Client Request Processing**: Client-side code might be modifying the received data
3. **Database Query Issue**: Wrong test record being retrieved
4. **Serialization Bug**: to_dict() method not working correctly

## Debug Steps Added

### Server-Side Debug (routes.py)
```python
logging.info(f"DEBUG: Sending test {test.id} to client {client_id}")
logging.info(f"DEBUG: Database test_type: {test.test_type}")
logging.info(f"DEBUG: Serialized test_type: {test_dict.get('test_type')}")
```

### Client-Side Debug (client.py)
```python
logger.info(f"DEBUG: Test type received: '{test_config.get('test_type')}' (type: {type(test_config.get('test_type'))})")
```

## Next Steps

1. **Create new WiFi Environmental Scan test** (Test 194)
2. **Check server logs** for debug output showing what's being sent
3. **Check client logs** for debug output showing what's being received
4. **Compare server vs client test_type values** to identify where override occurs

## Expected Resolution

The debug output will show either:
- Server sending wrong test_type (server-side bug)
- Client receiving correct test_type but processing it wrong (client-side bug)
- Database query returning wrong test record (query bug)

Once identified, the fix will be straightforward.