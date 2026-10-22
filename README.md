# 🎓 CourseWise AI – Personalized Course Recommender

Built for a **TCS hackathon**. CourseWise AI asks a student about their branch, semester, skills, interests, career goal and preferred difficulty, then recommends the **top 5 courses** that fit them best – with a score breakdown and interactive charts.

## ✨ Features

- Student profile form (name, branch, semester, skills, interests, career goal, difficulty)
- Top-5 personalised course recommendations with match scores
- Insights dashboard with Plotly charts in three tabs: **Course Demand**, **Skills**, **Careers**
- Clean, custom-themed Streamlit UI

## 🧠 How the recommender works

`recommender.py` combines text similarity with simple rule-based matches:

| Signal | Weight | How it's measured |
|--------|--------|-------------------|
| Content similarity | 30% | TF-IDF vectors of the student profile vs. course text, cosine similarity |
| Skills | 25% | Overlap between student skills and course skills |
| Interests | 20% | Interests found in the course description/category |
| Career | 15% | Student's career goal vs. the course's target career |
| Difficulty | 5% | How close the course level is to the preferred level |
| Rating | 5% | Course rating (normalised) |

The weighted total ranks all courses and the best 5 are returned by `recommend_courses(profile, courses)`.

## 📊 Dataset

`data/courses.csv` – 28 courses with these columns:

```
course_name, category, description, skills, career, difficulty, duration, rating, prerequisites
```

Add rows to this file to extend the catalogue – no code changes needed.

## 🛠️ Tech stack

- **Python**
- **Streamlit** (UI)
- **Pandas**, **NumPy**
- **scikit-learn** (`TfidfVectorizer`, `cosine_similarity`)
- **Plotly Express** (charts)

## 🚀 Run it

```bash
git clone https://github.com/Adarshmilan/TCShacakthon.git
cd TCShacakthon
pip install streamlit pandas numpy scikit-learn plotly
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

You can also test the engine on its own:

```bash
python recommender.py     # runs the built-in demo()
```
