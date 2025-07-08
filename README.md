# ai-price-fetcher

# 🔐 Bearer Token Authentication

### 🔸 How It Works
1. **Whitelisted Emails**  
   Set the environment variable:
   ```bash
   WHITELISTED_EMAILS="user1@example.com,user2@example.com"
   ```

2. **Generate Token**  
   Send a `POST` request to `/auth/token` with your email:
   ```bash
   curl -X POST "http://localhost:8000/auth/token" \
     -H "Content-Type: application/json" \
     -d '{"email": "admin@example.com"}'
   ```

3. **Use Token in API Requests**  
   Include it in the `Authorization` header:
   ```bash
   curl -X GET "http://localhost:8000/fetch-prices?product_name=iPhone%2015" \
     -H "Authorization: Bearer your-token-here"
   ```

---

# ⚡ In-Memory Caching

### ✅ Features
- **Automatic TTL**: Results cached for **30 minutes** by default  
- **Smart Keys**: Cache key is an **MD5 hash** of `product_name + country`  
- **Auto-Cleanup**: Expired items are cleaned up automatically  
- **Cache Management Endpoints**:
  - `GET /cache/stats` → View current cache stats  
  - `DELETE /cache/clear` → Clear all cached data  

---

# 🛡️ Security Features

- ⏳ **Token Expiration**: Default **24 hours** (`TOKEN_TTL`)  
- ✅ **Email Whitelisting**: Only approved emails can request tokens  
- 🔐 **Secure Hashing**: Uses **SHA-256** with your `AUTH_SECRET_KEY`  
- 🔒 **Protected Endpoints**: All `/fetch-prices` and `/cache/*` routes require a valid token  

---

# 📝 Environment Variables

```bash
# Required for authentication
WHITELISTED_EMAILS="admin@example.com,user@example.com"
AUTH_SECRET_KEY="your-super-secret-key-change-this"

# Optional settings
TOKEN_TTL="86400"  # 24 hours in seconds
OPENAI_API_KEY="your-openai-key"
```

---

# 🚀 API Endpoints

| Method | Endpoint            | Description                      | Auth Required |
|--------|---------------------|----------------------------------|---------------|
| POST   | `/auth/token`       | Generate bearer token            | ❌ No         |
| GET    | `/fetch-prices`     | Extract product prices           | ✅ Yes        |
| GET    | `/cache/stats`      | View cache statistics            | ✅ Yes        |
| DELETE | `/cache/clear`      | Clear the cache                  | ✅ Yes        |
| GET    | `/health`           | API health check                 | ❌ No         |

---

# 📊 Benefits

- ⚡ **Performance**: Cached results return instantly  
- 📉 **Rate Limiting**: Reduces dependency on external requests  
- 🔐 **Access Control**: Only whitelisted users can authenticate  
- 📈 **Monitoring**: Cache statistics and health checks  
- 🧠 **No Database**: Lightweight, **memory-only** design  

---

> ✅ The system is **production-ready**, secure, fast, and easy to maintain — with **authentication and in-memory caching** built-in!


https://github.com/user-attachments/assets/5c9dd8d6-e03f-4ab7-9039-b6c19f619988

> EXTENSIVE SWAGER DOCUMENTATION CAN BE FOUND: https://ai-price-fetcher.vercel.app/docs  
> NOTE: Added a simple Bearer Token auth just to save face(lol) - please dont miss this step


