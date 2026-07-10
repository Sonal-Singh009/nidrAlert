# 🚗 AI Drowsiness Detection System with Road Safety Chatbot

A fully functional web-based system that detects driver drowsiness in real-time using **MediaPipe**, stores alert data in **MongoDB**, and provides insights through **analytical dashboards**.
The platform also includes a **road safety chatbot** to assist users with safety-related queries.

---

## 🚀 Features

* 😴 Real-time drowsiness detection using webcam
* 🧠 Powered by **MediaPipe** for facial landmark tracking
* ⚠️ Instant alerts when drowsiness is detected
* 🗂 Stores driver alert sessions in **MongoDB**
* 📊 Analytical dashboard with driver performance insights
* 🤖 Integrated road safety chatbot
* 🌐 Fully functional web-based system

---

## 🧠 How It Works

1. Webcam captures live video feed
2. **MediaPipe** detects facial landmarks (eyes, mouth)
3. Eye state and facial patterns are analyzed
4. If drowsiness is detected:

   * Alert is triggered
   * Event is stored in database
5. Stored data is later used to:

   * Generate analytics
   * Track driver behavior over time

---

## 🛠 Tech Stack

* **Frontend:** (HTML, CSS, JavaScript / React )
* **Backend:** (Node.js / Flask / FastAPI)
* **AI/ML:** MediaPipe
* **Database:** MongoDB


---

## ⚙️ Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/drowsiness-detector.git
   ```

2. Navigate to the project folder:

   ```bash
   cd drowsiness-detector
   ```

3. Install dependencies:

   **For backend:**

   ```bash
   pip install -r requirements.txt
   ```

   OR

   ```bash
   npm install
   ```

4. Start the server:

   ```bash
   python main.py
   ```

   OR

   ```bash
   npm start
   ```

5. Open in browser:

   ```
   http://localhost:3000
   ```

---

## 📊 Analytics Dashboard

* Tracks drowsiness events per driver
* Visual representation using charts
* Helps analyze driving patterns and safety

---

## 🤖 Chatbot

* Provides road safety tips
* Answers user queries related to safe driving
* Enhances user engagement and awareness

---

## 📌 Future Improvements

* 📱 Mobile app integration
* 🔊 Voice alert system
* 🌙 Night-time detection optimization
* 🧠 Advanced ML models for higher accuracy
* ☁️ Cloud deployment & scalability

---

## 🙋‍♀️ Author

* Your Name

---

## 📌 Notes

* This project focuses on improving **road safety using AI**
* Combines real-time detection, backend systems, and analytics

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
