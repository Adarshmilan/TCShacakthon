"""
CourseWise AI
Personalized Course Recommendation Engine

Member 2 - Recommendation Engine

Technology:
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Cosine Similarity

Main function used by Streamlit:

    recommend_courses(profile, courses)

"""

import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

# Weights used for the final recommendation score.
#
# Total = 1.00
#
# Content similarity  -> 40%
# Skills              -> 25%
# Interests           -> 15%
# Career              -> 10%
# Difficulty           -> 5%
# Rating               -> 5%

CONTENT_WEIGHT = 0.30
SKILL_WEIGHT = 0.25
INTEREST_WEIGHT = 0.20
CAREER_WEIGHT = 0.15
DIFFICULTY_WEIGHT = 0.05
RATING_WEIGHT = 0.05


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_text(value):
    """
    Convert any value into normalized lowercase text.

    Example:
        "Machine Learning, Python"
        ->
        "machine learning python"
    """

    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        str(value).lower()
    ).strip()


def normalize_list(value):
    """
    Convert skills/interests into a clean list.

    Supports:

        ["Python", "SQL"]

    and:

        "Python, SQL"

    and:

        "Python;SQL"
    """

    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, list):
        items = value

    else:
        text = str(value)

        if "," in text:
            items = text.split(",")

        elif ";" in text:
            items = text.split(";")

        else:
            items = [text]

    return [
        normalize_text(item)
        for item in items
        if normalize_text(item)
    ]


def get_skill_match(student_skills, course_skills):
    """
    Calculate skill overlap between student and course.

    Returns:
        score between 0 and 1
    """

    student_skills = set(
        normalize_list(student_skills)
    )

    course_skills = set(
        normalize_list(course_skills)
    )

    if not course_skills:
        return 0.0

    if not student_skills:
        return 0.0

    # Direct overlap
    common = student_skills.intersection(course_skills)

    score = len(common) / len(course_skills)

    # Don't exceed 1
    return min(score, 1.0)


def get_interest_match(student_interests, course_text):
    """
    Calculate how strongly student interests
    appear in the course information.
    """

    interests = normalize_list(student_interests)

    if not interests:
        return 0.0

    course_text = normalize_text(course_text)

    matched = 0

    for interest in interests:

        if interest in course_text:
            matched += 1

    return matched / len(interests)


def get_career_match(student_career, course_career):
    """
    Calculate career relevance.

    Returns:
        1.0 -> strong match
        0.0 -> no match
    """

    student_career = normalize_text(student_career)
    course_career = normalize_text(course_career)

    if not student_career or not course_career:
        return 0.0

    # Exact / partial match
    if student_career in course_career:
        return 1.0

    if course_career in student_career:
        return 1.0

    # Keyword-based matching
    career_keywords = {

        "ai engineer": [
            "ai",
            "artificial intelligence",
            "machine learning",
            "deep learning",
            "nlp",
            "computer vision",
            "generative ai"
        ],

        "data scientist": [
            "data science",
            "machine learning",
            "statistics",
            "python",
            "analytics",
            "data"
        ],

        "full stack developer": [
            "web development",
            "frontend",
            "backend",
            "full stack",
            "javascript",
            "react",
            "node"
        ],

        "cyber security engineer": [
            "cyber security",
            "security",
            "network security",
            "ethical hacking",
            "penetration testing"
        ],

        "cloud engineer": [
            "cloud",
            "aws",
            "azure",
            "gcp",
            "devops"
        ],

        "devops engineer": [
            "devops",
            "docker",
            "kubernetes",
            "ci cd",
            "cloud",
            "automation"
        ]
    }

    keywords = career_keywords.get(
        student_career,
        []
    )

    if not keywords:
        return 0.0

    matched = sum(
        1
        for keyword in keywords
        if keyword in course_career
    )

    return min(
        matched / max(len(keywords), 1),
        1.0
    )


def get_difficulty_match(student_difficulty, course_difficulty):
    """
    Match student preferred difficulty with course difficulty.

    Exact match       -> 1.0
    One level away    -> 0.7
    Two levels away   -> 0.3
    """

    levels = {
        "beginner": 0,
        "intermediate": 1,
        "advanced": 2
    }

    student = normalize_text(
        student_difficulty
    )

    course = normalize_text(
        course_difficulty
    )

    if student not in levels or course not in levels:
        return 0.5

    difference = abs(
        levels[student] - levels[course]
    )

    if difference == 0:
        return 1.0

    if difference == 1:
        return 0.7

    return 0.3


def get_rating_score(rating):
    """
    Convert course rating into 0-1 score.

    Assumes rating is out of 5.
    """

    try:
        rating = float(rating)
    except (ValueError, TypeError):
        return 0.5

    return min(
        max(rating / 5.0, 0.0),
        1.0
    )


# ============================================================
# COURSE TEXT
# ============================================================

def create_course_text(course):
    """
    Combine important course information into
    one searchable text representation.

    This text is used by TF-IDF.
    """

    fields = [
        course.get("course_name", ""),
        course.get("name", ""),
        course.get("category", ""),
        course.get("description", ""),
        course.get("skills", ""),
        course.get("career", ""),
        course.get("prerequisites", "")
    ]

    return " ".join(
        normalize_text(field)
        for field in fields
    )


# ============================================================
# STUDENT PROFILE TEXT
# ============================================================

def create_student_text(profile):
    """
    Convert student profile into searchable text.
    """

    skills = " ".join(
        normalize_list(
            profile.get("skills", [])
        )
    )

    interests = " ".join(
        normalize_list(
            profile.get("interests", [])
        )
    )

    career = normalize_text(
        profile.get("career", "")
    )

    branch = normalize_text(
        profile.get("branch", "")
    )

    return " ".join(
        [
            skills,
            interests,
            career,
            branch
        ]
    )


# ============================================================
# MAIN RECOMMENDATION FUNCTION
# ============================================================

def recommend_courses(
    profile,
    courses,
    top_n=5
):
    """
    Generate personalized course recommendations.

    Parameters
    ----------
    profile : dict
        Student information.

    courses : pandas.DataFrame OR list of dictionaries
        Course dataset.

    top_n : int
        Number of recommendations.

    Returns
    -------
    list of dictionaries

        Each recommendation contains:

        name
        category
        match
        difficulty
        duration
        rating
        reason
    """

    # --------------------------------------------------------
    # Validate profile
    # --------------------------------------------------------

    if profile is None:
        profile = {}

    # --------------------------------------------------------
    # Convert courses to DataFrame
    # --------------------------------------------------------

    if isinstance(courses, pd.DataFrame):

        df = courses.copy()

    elif isinstance(courses, list):

        df = pd.DataFrame(courses)

    else:

        raise TypeError(
            "courses must be a pandas DataFrame "
            "or a list of dictionaries."
        )

    # --------------------------------------------------------
    # Check empty dataset
    # --------------------------------------------------------

    if df.empty:
        return []

    # --------------------------------------------------------
    # Make sure required columns exist
    # --------------------------------------------------------

    optional_columns = {
        "course_name": "",
        "name": "",
        "category": "",
        "description": "",
        "skills": "",
        "career": "",
        "difficulty": "Intermediate",
        "duration": "Self-paced",
        "rating": 4.0,
        "prerequisites": ""
    }

    for column, default_value in optional_columns.items():

        if column not in df.columns:

            df[column] = default_value

    # --------------------------------------------------------
    # Create course text
    # --------------------------------------------------------

    df["_course_text"] = df.apply(
        create_course_text,
        axis=1
    )

    # --------------------------------------------------------
    # Student text
    # --------------------------------------------------------

    student_text = create_student_text(
        profile
    )

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    all_text = (
        df["_course_text"].tolist()
        + [student_text]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:

        vectors = vectorizer.fit_transform(
            all_text
        )

        course_vectors = vectors[:-1]

        student_vector = vectors[-1]

        similarities = cosine_similarity(
            student_vector,
            course_vectors
        )[0]

    except ValueError:

        # Fallback if dataset contains no useful text
        similarities = np.zeros(
            len(df)
        )

    # --------------------------------------------------------
    # Student information
    # --------------------------------------------------------

    student_skills = profile.get(
        "skills",
        []
    )

    student_interests = profile.get(
        "interests",
        []
    )

    student_career = profile.get(
        "career",
        ""
    )

    student_difficulty = profile.get(
        "difficulty",
        "Intermediate"
    )

    # --------------------------------------------------------
    # Calculate scores
    # --------------------------------------------------------

    results = []

    for index, (_, course) in enumerate(
        df.iterrows()
    ):

        content_score = float(
            similarities[index]
        )

        # Skill match
        skill_score = get_skill_match(
            student_skills,
            course["skills"]
        )

        # Interest match
        interest_score = get_interest_match(
            student_interests,
            " ".join(
                [
                    str(course["category"]),
                    str(course["description"]),
                    str(course["skills"])
                ]
            )
        )

        # Career match
        career_score = get_career_match(
            student_career,
            course["career"]
        )

        # Difficulty match
        difficulty_score = get_difficulty_match(
            student_difficulty,
            course["difficulty"]
        )

        # Rating
        rating_score = get_rating_score(
            course["rating"]
        )

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        final_score = (

            CONTENT_WEIGHT
            * content_score

            +

            SKILL_WEIGHT
            * skill_score

            +

            INTEREST_WEIGHT
            * interest_score

            +

            CAREER_WEIGHT
            * career_score

            +

            DIFFICULTY_WEIGHT
            * difficulty_score

            +

            RATING_WEIGHT
            * rating_score

        )

        # Convert to percentage
        match_percentage = round(
            min(
                max(
                    final_score * 100,
                    0
                ),
                100
            )
        )

        # ----------------------------------------------------
        # Recommendation explanation
        # ----------------------------------------------------

        reasons = []

        if content_score >= 0.20:
            reasons.append(
                "strongly matches your learning profile"
            )

        if skill_score >= 0.25:
            reasons.append(
                "matches your existing skills"
            )

        if interest_score >= 0.30:
            reasons.append(
                "matches your interests"
            )

        if career_score >= 0.20:
            reasons.append(
                "supports your career goal"
            )

        if difficulty_score >= 0.70:
            reasons.append(
                "fits your preferred difficulty"
            )

        if not reasons:
            reasons.append(
                "provides useful skills for your learning journey"
            )

        reason = (
            reasons[0].capitalize()
            + (
                ", " + ", ".join(reasons[1:])
                if len(reasons) > 1
                else ""
            )
            + "."
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        course_name = str(
            course["course_name"]
        ).strip()

        if not course_name:

            course_name = str(
                course["name"]
            ).strip()

        result = {
            "name": course_name,

            "category": str(
                course["category"]
            ),

            "match": match_percentage,

            "difficulty": str(
                course["difficulty"]
            ),

            "duration": str(
                course["duration"]
            ),

            "rating": float(
                course["rating"]
            ),

            "reason": reason,

            # Additional ML information
            "content_score": round(
                content_score,
                3
            ),

            "skill_score": round(
                skill_score,
                3
            ),

            "interest_score": round(
                interest_score,
                3
            ),

            "career_score": round(
                career_score,
                3
            )
        }

        results.append(result)

    # --------------------------------------------------------
    # Sort by recommendation score
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["match"],
        reverse=True
    )

    # --------------------------------------------------------
    # Return top N
    # --------------------------------------------------------

    return results[:top_n]


# ============================================================
# LOAD COURSE DATASET
# ============================================================

def load_courses(csv_path="data/courses.csv"):
    """
    Load course dataset from CSV.
    """

    try:

        df = pd.read_csv(
            csv_path
        )

        return df

    except FileNotFoundError:

        print(
            f"Course dataset not found: {csv_path}"
        )

        return pd.DataFrame()


# ============================================================
# TEST FUNCTION
# ============================================================

def demo():

    print("\n")
    print("=" * 60)
    print("       COURSEWISE AI - RECOMMENDATION ENGINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Example student
    # --------------------------------------------------------

    student = {

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

    # --------------------------------------------------------
    # Load courses
    # --------------------------------------------------------

    courses = load_courses()

    if courses.empty:

        print(
            "\n❌ No course dataset found."
        )

        print(
            "Create data/courses.csv first."
        )

        return

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    recommendations = recommend_courses(
        student,
        courses,
        top_n=5
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n")
    print("🎯 PERSONALIZED RECOMMENDATIONS")
    print("-" * 60)

    for index, course in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"\n{index}. {course['name']}"
        )

        print(
            f"   Match: {course['match']}%"
        )

        print(
            f"   Category: {course['category']}"
        )

        print(
            f"   Difficulty: {course['difficulty']}"
        )

        print(
            f"   Rating: ⭐ {course['rating']}"
        )

        print(
            f"   Why: {course['reason']}"
        )

    print("\n")
    print("=" * 60)


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    demo()