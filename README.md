# 🍴 Swiggy’s Restaurant Recommendation System using Streamlit

## 📌 Overview
This project builds a **restaurant recommendation system** inspired by Swiggy, using **Python, scikit-learn, and Streamlit**. The system recommends restaurants based on user preferences such as city, cuisine, rating, and cost. It leverages clustering and similarity-based methods to generate personalized suggestions, displayed through an interactive Streamlit interface.

---

## 🎯 Skills You’ll Gain
- Data Cleaning & Preprocessing  
- One-Hot Encoding of categorical features  
- Clustering (K-Means, Cosine Similarity, etc.)  
- Streamlit Application Development  
- Python Programming  

---

## 🛠️ Tech Stack
- **Python**  
- **Pandas / NumPy**  
- **Scikit-learn**  
- **Streamlit**  

---

## 📂 Dataset
The dataset contains the following columns:


- **Categorical**: name, city, cuisine  
- **Numerical**: rating, rating_count, cost  

---

## 🚀 Approach
1. **Data Cleaning**
   - Remove duplicates  
   - Handle missing values  
   - Save cleaned dataset (`cleaned_data.csv`)  

2. **Data Preprocessing**
   - Apply One-Hot Encoding to categorical features  
   - Save encoder (`encoder.pkl`)  
   - Ensure indices align between cleaned and encoded datasets  

3. **Recommendation Engine**
   - Use **K-Means clustering** or **Cosine Similarity**  
   - Map results back to cleaned dataset for interpretation  

4. **Streamlit Application**
   - Accept user inputs (city, cuisine, rating, price)  
   - Generate recommendations  
   - Display results in a user-friendly interface  

---

## 📊 Business Use Cases
- **Personalized Recommendations**: Help users discover restaurants tailored to their tastes.  
- **Improved Customer Experience**: Enhance decision-making with relevant suggestions.  
- **Market Insights**: Understand customer preferences for targeted marketing.  
- **Operational Efficiency**: Optimize offerings based on popular demand.  

---

## 📸 Demo
Run the app locally:
```bash
streamlit run streamlit_app.py





