# Deploy Mental Health Chatbot (Free)

Get a **direct link** you can share with anyone.

---

## Option A: Railway (Recommended - Often Works Better)

1. Push your code to GitHub (see Step 1 below).
2. Go to [railway.app](https://railway.app) and sign up with GitHub.
3. Click **New Project** → **Deploy from GitHub repo** → select `mental-health-chatbot`.
4. After deploy starts, click your service → **Variables** → add:
   - `OPENAI_API_KEY` = your OpenAI API key
5. Click **Settings** → **Generate Domain** to get your public URL.
6. Your link: `https://your-app.up.railway.app`

---

## Option B: Render

### Step 1: Push to GitHub

```powershell
cd c:\Users\ASUS\Downloads\mental-health-ai-chatbot
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mental-health-chatbot.git
git push -u origin main
```

### Step 2: Deploy on Render

1. Go to [render.com](https://render.com) → **New** → **Web Service**.
2. Connect GitHub and select `mental-health-chatbot`.
3. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn -c gunicorn_config.py app:app`
4. **Environment** → Add `OPENAI_API_KEY` = your key.
5. **Create Web Service**.

### If Render Gets Stuck at "Almost Live"

1. **Manual redeploy:** Dashboard → your service → **Manual Deploy** → **Deploy latest commit**.
2. **Check Logs:** Open **Logs** tab - look for Python errors. Copy any error message.
3. **Try Railway:** Use Option A above instead - it often deploys faster.

---

## Notes

- **Free tier:** Apps may sleep after inactivity. First load can take 30-60 seconds.
- **OpenAI:** Get an API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
