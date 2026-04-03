# AgriAI Backend

This is the backend server for the **AgriAI** ecosystem, providing intelligent crop disease classification over SMS. 
It receives image embeddings sent from an Android client via a Twilio SMS Webhook, runs Principal Component Analysis (PCA) and an XGBoost Model to predict crop diseases, and returns the confidence level back via SMS.

## Technologies Used
- **Python / Flask** (Backend API framework)
- **Twilio** (SMS gateway integration)
- **Firebase Admin** (Firestore database for rate limiting & logging feedback)
- **XGBoost & Scikit-Learn** (Machine learning prediction stack)
- **Gunicorn** (Production web server)

---

## 🚀 Running Locally

1. Create a Python virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r firebase_backend/requirements.txt
   ```
3. Set your internal environment variables. Create a `.env` file with:
   ```env
   TWILIO_SID=your_twilio_sid
   TWILIO_TOKEN=your_twilio_token
   AGRIAI_NUMBER=1800AGRIAI
   ```
4. Start the server (for local testing):
   ```bash
   cd firebase_backend
   flask --app main run
   ```

---

## ☁️ Deployment (Render)

This repository is optimized for free deployment on [Render](https://render.com).

1. Connect your GitHub repository as a new **Web Service** on Render.
2. In the Render setup, configure the following:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r firebase_backend/requirements.txt`
   - **Start Command:** `cd firebase_backend && gunicorn main:app`
3. Under **Environment Variables** in Render, securely add your `TWILIO_SID` and `TWILIO_TOKEN`.
4. Deploy!
5. Once your application is live, grab your `https://your-service.onrender.com` URL and paste it with `/sms` mapped at the end (`...onrender.com/sms`) into your Twilio Console Webhook settings.
