# Fix: Website Loading Forever / Not Opening

Render's free tier **puts your app to sleep** after ~15 minutes of no traffic. The first visit after that can take **1-2 minutes** to load.

---

## Quick Fix 1: Wait Longer

1. Open: https://mental-health-chatbot89.onrender.com
2. **Wait 1-2 minutes** (do not close the tab)
3. The page should load

---

## Quick Fix 2: Keep It Awake (Recommended)

Use **UptimeRobot** (free) to ping your site every 5 minutes:

1. Go to [uptimerobot.com](https://uptimerobot.com) and sign up (free)
2. Click **Add New Monitor**
3. **Monitor Type:** HTTP(s)
4. **Friendly Name:** Mental Health Chatbot
5. **URL:** https://mental-health-chatbot89.onrender.com/health
6. **Monitoring Interval:** 5 minutes
7. Click **Create Monitor**

Your site will stay awake and load quickly.

---

## Quick Fix 3: Check Render Dashboard

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click your **mental-health-chatbot89** service
3. Check **Logs** tab - any red errors?
4. Status should be **Live** (green)

---

## If Still Not Working

Try **Railway** instead (often faster):

1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub → select mental-health-chatbot89
3. Add GEMINI_API_KEY in Variables
4. Generate Domain
5. Railway free tier often has faster cold starts
