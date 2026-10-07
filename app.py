import streamlit as st
import pandas as pd
import plotly.express as px

from recommender import recommend_courses, load_courses


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CourseWise AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL UI THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APPLICATION
       ======================================================== */

    .stApp {
        background-color: #f4f7fb !important;
    }

    .main {
        background-color: #f4f7fb !important;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* ========================================================
       FORCE MAIN TEXT TO DARK COLORS
       ======================================================== */

    .main p,
    .main span,
    .main label,
    .main li {
        color: #334155 !important;
    }

    .main h1,
    .main h2,
    .main h3,
    .main h4,
    .main h5,
    .main h6 {
        color: #0f172a !important;
    }

    .stMarkdown {
        color: #334155 !important;
    }

    .stMarkdown p {
        color: #475569 !important;
    }

    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown h5,
    .stMarkdown h6 {
        color: #0f172a !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0f172a 0%,
            #172554 55%,
            #1e1b4b 100%
        ) !important;

        border-right: 1px solid #26324a;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #cbd5e1 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #ffffff !important;
    }


    /* ========================================================
       PAGE TITLES
       ======================================================== */

    .page-title {
        color: #0f172a !important;
        font-size: 2.7rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 0.35rem;
    }

    .page-subtitle {
        color: #64748b !important;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 2rem;
    }

    .section-title {
        color: #0f172a !important;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero-title {
        color: #0f172a !important;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .hero-description {
        color: #475569 !important;
        font-size: 1.05rem;
        line-height: 1.7;
    }


    /* ========================================================
       CONTAINERS
       ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #dbe3ef !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.045);
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #dbe3ef !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    div[data-testid="stMetricLabel"] * {
        color: #64748b !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 800 !important;
    }

    div[data-testid="stMetricValue"] * {
        color: #0f172a !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #16a34a !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
        border: 1px solid #4f46e5 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 45px;
        transition: 0.2s ease;
    }

    .stButton > button p {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        background-color: #4338ca !important;
        border-color: #4338ca !important;
        color: #ffffff !important;
    }


    /* ========================================================
       INPUT LABELS
       ======================================================== */

    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label,
    .stMultiSelect label,
    .stRadio label {
        color: #334155 !important;
        font-weight: 650 !important;
    }


    /* ========================================================
       INPUT BOXES
       ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] span {
        color: #334155 !important;
    }

    input {
        color: #0f172a !important;
        background-color: #ffffff !important;
    }

    textarea {
        color: #0f172a !important;
        background-color: #ffffff !important;
    }


    /* ========================================================
       ALERT BOXES
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    div[data-testid="stAlert"] p {
        color: #334155 !important;
    }


    /* ========================================================
       INFO BOX
       ======================================================== */

    div[data-testid="stAlert"][kind="info"] p {
        color: #1e3a8a !important;
    }


    /* ========================================================
       TABS
       ======================================================== */

    button[data-baseweb="tab"] {
        color: #64748b !important;
        font-weight: 650 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4f46e5 !important;
    }


    /* ========================================================
       CHAT
       ======================================================== */

    [data-testid="stChatMessage"] {
        border-radius: 14px !important;
        border: 1px solid #e2e8f0;
    }

    [data-testid="stChatMessage"] p {
        color: #334155 !important;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: #e2e8f0 !important;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        color: #94a3b8 !important;
        font-size: 0.85rem;
        padding-top: 35px;
        padding-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

courses = load_courses("data/courses.csv")


# ============================================================
# FALLBACK DATA
# ============================================================

if courses.empty:

    courses = pd.DataFrame([
        {
            "course_name": "Machine Learning with Python",
            "category": "Artificial Intelligence",
            "description": "Machine learning using Python",
            "skills": "Python, Machine Learning, Statistics",
            "career": "AI Engineer",
            "difficulty": "Intermediate",
            "duration": "8 weeks",
            "rating": 4.8,
            "prerequisites": "Python"
        },
        {
            "course_name": "Deep Learning Fundamentals",
            "category": "Artificial Intelligence",
            "description": "Neural networks and deep learning",
            "skills": "Python, Deep Learning, Neural Networks",
            "career": "AI Engineer",
            "difficulty": "Advanced",
            "duration": "10 weeks",
            "rating": 4.7,
            "prerequisites": "Machine Learning"
        },
        {
            "course_name": "Natural Language Processing",
            "category": "Artificial Intelligence",
            "description": "Natural language processing",
            "skills": "Python, NLP, Machine Learning",
            "career": "AI Engineer",
            "difficulty": "Intermediate",
            "duration": "8 weeks",
            "rating": 4.6,
            "prerequisites": "Machine Learning"
        },
        {
            "course_name": "Data Science with Python",
            "category": "Data Science",
            "description": "Python data science and analytics",
            "skills": "Python, Pandas, Statistics",
            "career": "Data Scientist",
            "difficulty": "Intermediate",
            "duration": "7 weeks",
            "rating": 4.7,
            "prerequisites": "Python"
        }
    ])


# ============================================================
# SESSION STATE
# ============================================================

if "profile" not in st.session_state:

    st.session_state.profile = {
        "name": "Adarsh",
        "branch": "Computer Science",
        "semester": 5,
        "gpa": 8.6,
        "skills": [
            "Python",
            "Java",
            "SQL"
        ],
        "interests": [
            "Artificial Intelligence",
            "Machine Learning"
        ],
        "career": "AI Engineer",
        "difficulty": "Intermediate"
    }


if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! 👋 I'm your CourseWise AI assistant. "
                "Ask me about courses, career paths, skills, "
                "or your learning roadmap."
            )
        }
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 CourseWise AI")

    st.caption(
        "Personalized University Learning Advisor"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "👤 Student Profile",
            "🎯 Recommendations",
            "📊 Analytics",
            "🤖 AI Assistant"
        ]
    )

    st.markdown("---")

    profile = st.session_state.profile

    st.markdown("### 👤 Current Student")

    st.markdown(
        f"**{profile['name']}**"
    )

    st.caption(
        f"{profile['branch']} • Semester {profile['semester']}"
    )

    st.caption(
        f"🎯 {profile['career']}"
    )

    st.markdown("---")

    st.caption(
        "CourseWise AI"
    )

    st.caption(
        "Hackathon Prototype • 2026"
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="page-title">🎓 CourseWise AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Your intelligent learning companion for personalized '
        'course discovery and career readiness.'
        '</div>',
        unsafe_allow_html=True
    )

    # HERO

    with st.container(border=True):

        st.markdown(
            "### 🚀 Learn smarter. Build your career faster."
        )

        st.write(
            "CourseWise AI analyzes your academic profile, "
            "skills, interests and career aspirations to "
            "recommend the courses that matter most for you."
        )

        st.markdown("")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.success(
                "🎯 Personalized Recommendations"
            )

        with c2:
            st.info(
                "🧠 Skill Gap Analysis"
            )

        with c3:
            st.warning(
                "🗺️ Career Learning Paths"
            )

    # PLATFORM OVERVIEW

    st.markdown(
        '<div class="section-title">📈 Platform Overview</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "📚 Courses Available",
            len(courses)
        )

    with c2:
        st.metric(
            "👨‍🎓 Students",
            "1,240",
            "+18%"
        )

    with c3:
        st.metric(
            "🎯 Average Match",
            "86%"
        )

    with c4:
        st.metric(
            "⭐ Average Rating",
            "4.6"
        )

    # FEATURES

    st.markdown(
        '<div class="section-title">✨ Platform Capabilities</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        with st.container(border=True):

            st.markdown("### 🎯 Recommendations")

            st.write(
                "AI-powered course ranking based on "
                "skills, interests, career goals and "
                "learning preferences."
            )

            st.success(
                "Personalized"
            )

    with c2:

        with st.container(border=True):

            st.markdown("### 🧠 Skill Gap")

            st.write(
                "Identify missing skills and discover "
                "what you should learn next for your "
                "target career."
            )

            st.info(
                "Actionable Insights"
            )

    with c3:

        with st.container(border=True):

            st.markdown("### 🗺️ Career Path")

            st.write(
                "Build a structured roadmap from your "
                "current knowledge to career-ready skills."
            )

            st.warning(
                "Career Focused"
            )

    st.markdown("")

    if st.button(
        "🚀 Start My Personalized Learning Journey",
        type="primary",
        use_container_width=True
    ):

        st.info(
            "👉 Open **Student Profile** from the sidebar "
            "to create your personalized plan."
        )


# ============================================================
# STUDENT PROFILE
# ============================================================

elif page == "👤 Student Profile":

    profile = st.session_state.profile

    st.markdown(
        '<div class="page-title">👤 Student Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Build your learning profile so CourseWise AI can '
        'generate relevant recommendations.'
        '</div>',
        unsafe_allow_html=True
    )

    # ACADEMIC

    st.markdown(
        '<div class="section-title">📋 Academic Information</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        c1, c2 = st.columns(2)

        with c1:

            name = st.text_input(
                "Student Name",
                value=profile["name"]
            )

            branches = [
                "Computer Science",
                "Information Technology",
                "Electronics",
                "Mechanical",
                "Civil"
            ]

            branch = st.selectbox(
                "Branch",
                branches,
                index=branches.index(
                    profile["branch"]
                )
            )

        with c2:

            semester = st.selectbox(
                "Current Semester",
                list(range(1, 9)),
                index=profile["semester"] - 1
            )

            gpa = st.number_input(
                "Current GPA",
                min_value=0.0,
                max_value=10.0,
                value=float(profile["gpa"]),
                step=0.1
            )

    # SKILLS

    st.markdown(
        '<div class="section-title">💻 Skills & Interests</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        c1, c2 = st.columns(2)

        with c1:

            skills = st.multiselect(
                "Current Skills",
                [
                    "Python",
                    "Java",
                    "C",
                    "JavaScript",
                    "TypeScript",
                    "SQL",
                    "React",
                    "Machine Learning",
                    "Data Science",
                    "HTML/CSS",
                    "Cloud Computing",
                    "Git",
                    "Docker",
                    "Linux"
                ],
                default=profile["skills"]
            )

        with c2:

            interests = st.multiselect(
                "Areas of Interest",
                [
                    "Artificial Intelligence",
                    "Machine Learning",
                    "Data Science",
                    "Web Development",
                    "Cyber Security",
                    "Cloud Computing",
                    "Mobile Development",
                    "DevOps",
                    "Generative AI"
                ],
                default=profile["interests"]
            )

    # CAREER

    st.markdown(
        '<div class="section-title">🎯 Career Goal</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        careers = [
            "AI Engineer",
            "Data Scientist",
            "Full Stack Developer",
            "Cyber Security Engineer",
            "Cloud Engineer",
            "DevOps Engineer",
            "Software Developer"
        ]

        career = st.selectbox(
            "Target Career",
            careers,
            index=(
                careers.index(profile["career"])
                if profile["career"] in careers
                else 0
            )
        )

        difficulties = [
            "Beginner",
            "Intermediate",
            "Advanced"
        ]

        difficulty = st.radio(
            "Preferred Course Difficulty",
            difficulties,
            index=difficulties.index(
                profile["difficulty"]
            ),
            horizontal=True
        )

    st.markdown("")

    if st.button(
        "💾 Save Profile & Generate Recommendations",
        type="primary",
        use_container_width=True
    ):

        st.session_state.profile = {
            "name": name,
            "branch": branch,
            "semester": semester,
            "gpa": gpa,
            "skills": skills,
            "interests": interests,
            "career": career,
            "difficulty": difficulty
        }

        st.success(
            "✅ Profile saved successfully!"
        )

        st.info(
            "🎯 Open **Recommendations** to see your personalized courses."
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "🎯 Recommendations":

    profile = st.session_state.profile

    st.markdown(
        '<div class="page-title">🎯 Personalized Recommendations</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'AI-powered courses selected specifically for your profile.'
        '</div>',
        unsafe_allow_html=True
    )

    # PROFILE SUMMARY

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "👤 Student",
            profile["name"]
        )

    with c2:
        st.metric(
            "🎓 Semester",
            profile["semester"]
        )

    with c3:
        st.metric(
            "📊 GPA",
            profile["gpa"]
        )

    with c4:
        st.metric(
            "🎯 Career",
            profile["career"]
        )

    st.markdown("")

    # RECOMMENDATIONS

    recommendations = recommend_courses(
        profile,
        courses,
        top_n=5
    )

    st.markdown(
        '<div class="section-title">🏆 Top Course Recommendations</div>',
        unsafe_allow_html=True
    )

    for index, course in enumerate(
        recommendations,
        start=1
    ):

        with st.container(border=True):

            left, right = st.columns([5, 1])

            with left:

                st.markdown(
                    f"### {index}. {course['name']}"
                )

                st.caption(
                    f"{course['category']}  •  "
                    f"{course['difficulty']}  •  "
                    f"{course['duration']}"
                )

                st.write(
                    f"⭐ **{course['rating']} / 5.0**"
                )

                st.info(
                    f"💡 **Why recommended?**\n\n"
                    f"{course['reason']}"
                )

            with right:

                st.metric(
                    "Match",
                    f"{course['match']}%"
                )

    # SKILL GAP

    st.markdown(
        '<div class="section-title">🧠 Skill Gap Analysis</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.write(
            f"Target career: **{profile['career']}**"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "🐍 Python",
                "Strong"
            )

        with c2:
            st.metric(
                "🤖 Machine Learning",
                "Good"
            )

        with c3:
            st.metric(
                "🧠 Deep Learning",
                "Priority"
            )

        with c4:
            st.metric(
                "💬 NLP",
                "Priority"
            )

        st.warning(
            "🎯 Recommended next skill: Deep Learning"
        )

    # ROADMAP

    st.markdown(
        '<div class="section-title">🗺️ Recommended Career Roadmap</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        roadmap = [
            ("1", "Python", "Foundation"),
            ("2", "Statistics", "Core"),
            ("3", "Machine Learning", "Core AI"),
            ("4", "Deep Learning", "Advanced AI"),
            ("5", "Natural Language Processing", "Specialization"),
            ("6", "Generative AI", "Emerging Technology")
        ]

        for number, title, category in roadmap:

            c1, c2, c3 = st.columns(
                [1, 5, 2]
            )

            with c1:
                st.markdown(
                    f"### {number}"
                )

            with c2:
                st.write(
                    f"**{title}**"
                )

            with c3:
                st.caption(
                    category
                )

            if number != "6":
                st.divider()


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="page-title">📊 Learning Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Explore student interests, skills and career preferences.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "👨‍🎓 Students",
            "1,240",
            "+18%"
        )

    with c2:
        st.metric(
            "📚 Courses",
            len(courses)
        )

    with c3:
        st.metric(
            "⭐ Avg Rating",
            "4.6"
        )

    with c4:
        st.metric(
            "🎯 Avg Match",
            "86%"
        )

    st.markdown("")

    tab1, tab2, tab3 = st.tabs(
        [
            "📈 Course Demand",
            "💻 Skills",
            "🎯 Careers"
        ]
    )

    # COURSE DEMAND

    with tab1:

        data = pd.DataFrame({
            "Category": [
                "AI",
                "Data Science",
                "Web Development",
                "Cyber Security",
                "Cloud",
                "DevOps"
            ],
            "Students": [
                320,
                280,
                250,
                190,
                200,
                170
            ]
        })

        fig = px.bar(
            data,
            x="Category",
            y="Students",
            text="Students",
            title="Student Interest by Course Category"
        )

        fig.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(
                color="#334155"
            ),
            title_font=dict(
                color="#0f172a"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # SKILLS

    with tab2:

        data = pd.DataFrame({
            "Skill": [
                "Python",
                "SQL",
                "Java",
                "Machine Learning",
                "Cloud",
                "React"
            ],
            "Students": [
                850,
                620,
                540,
                430,
                380,
                350
            ]
        })

        fig = px.bar(
            data,
            x="Skill",
            y="Students",
            text="Students",
            title="Most Common Student Skills"
        )

        fig.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(
                color="#334155"
            ),
            title_font=dict(
                color="#0f172a"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # CAREERS

    with tab3:

        data = pd.DataFrame({
            "Career": [
                "AI Engineer",
                "Data Scientist",
                "Full Stack Developer",
                "Cyber Security",
                "Cloud Engineer",
                "DevOps"
            ],
            "Students": [
                300,
                260,
                320,
                180,
                180,
                150
            ]
        })

        fig = px.pie(
            data,
            names="Career",
            values="Students",
            title="Student Career Preferences"
        )

        fig.update_layout(
            paper_bgcolor="#ffffff",
            font=dict(
                color="#334155"
            ),
            title_font=dict(
                color="#0f172a"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.info(
        "📌 Analytics uses simulated university data "
        "for this hackathon prototype."
    )


# ============================================================
# AI ASSISTANT
# ============================================================

elif page == "🤖 AI Assistant":

    profile = st.session_state.profile

    st.markdown(
        '<div class="page-title">🤖 AI Learning Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Your conversational guide for courses, skills and career planning.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.container(border=True):

        st.markdown(
            f"### 👋 Hello, {profile['name']}!"
        )

        st.write(
            f"You're currently targeting a career as a "
            f"**{profile['career']}**."
        )

        st.write(
            "Ask me about courses, skill gaps, career paths "
            "or your learning roadmap."
        )

    st.markdown("")

    # CHAT

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    prompt = st.chat_input(
        "Ask: What should I learn after Machine Learning?"
    )

    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        question = prompt.lower()

        # TEMPORARY FALLBACK
        # Member 4 can replace this with Ollama.

        if (
            "ai" in question
            or "machine learning" in question
            or "artificial intelligence" in question
        ):

            response = """
### 🤖 AI Engineer Learning Path

Based on your profile:

1. 🐍 Python
2. 📊 Statistics
3. 🤖 Machine Learning
4. 🧠 Deep Learning
5. 💬 NLP
6. ✨ Generative AI

**Recommended next step:** Deep Learning.
"""

        elif "web" in question:

            response = """
### 🌐 Full Stack Development Path

1. HTML & CSS
2. JavaScript
3. React
4. Node.js
5. Databases
6. REST APIs
7. Cloud Deployment
"""

        elif "data" in question:

            response = """
### 📊 Data Science Path

1. Python
2. Statistics
3. Pandas
4. SQL
5. Data Visualization
6. Machine Learning
"""

        elif "course" in question:

            response = """
### 📚 Recommended Courses

Based on your current profile:

🥇 Machine Learning with Python

🥈 Deep Learning Fundamentals

🥉 Natural Language Processing

These courses strongly support your career goal.
"""

        elif "skill" in question:

            response = """
### 🧠 Your Skill Gap

Current strengths:

✅ Python
✅ SQL
✅ Java

Priority skills:

⚡ Deep Learning
⚡ NLP
⚡ Generative AI

Start with Deep Learning.
"""

        else:

            response = f"""
Based on your goal of becoming a **{profile['career']}**,
I recommend courses that strengthen your existing skills
while closing your biggest skill gaps.

Try asking:

• What should I learn for AI?
• Recommend courses
• What skills am I missing?
• Give me a data science roadmap
"""

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        with st.chat_message("assistant"):
            st.markdown(response)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        🎓 CourseWise AI • Personalized University Learning Advisor
        <br>
        Hackathon Prototype • Python • Streamlit • Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)