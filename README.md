#  Cardiovascular Health Report App

A Streamlit dashboard that generates a personalized cardiovascular health report using a trained Gradient Boosting model.

---

##  Features
- Sidebar input form for patient details.
- Dashboard layout with:
  -  Input summary table
  - Risk probability gauge
  -  Personalized recommendations
- Gradient Boosting model (`gradient_boosting_model.pkl`) for prediction.

---

## Project Structure
cardio-risk-app/
├── app.py                   # Streamlit app script
├── requirements.txt         # Dependencies
└── gradient_boosting_model.pkl  # Trained model file

---

##  Installation (Local Run)
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cardio-risk-app.git
   cd cardio-risk-app
pip install -r requirements.txt


streamlit run app.py
Deployment
 Deployment This app is deployed on Streamlit Cloud: Live App Link:https://youssefrabeay22-cardio-risk-app-app-rpvkse.streamlit.app/
