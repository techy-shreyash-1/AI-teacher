import os
import requests
import feedparser
from urllib.parse import quote
from datetime import datetime
from flask import Flask, render_template_string

app = Flask(__name__)

# =========================================================
# CONFIGURATION
# =========================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gpt-oss:120b-cloud"

RSS_URL = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"

# =========================================================
# HTML + CSS FRONTEND
# =========================================================

HTML = """
<!DOCTYPE html>
<html lang="mr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>AI Marathi News</title>

    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: Arial, sans-serif;
            background: #f4f6f9;
            color: #222;
        }

        header {
            background: #111827;
            color: white;
            padding: 20px;
        }

        .header-content {
            max-width: 1200px;
            margin: auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            font-size: 26px;
            font-weight: bold;
        }

        .logo span {
            color: #fbbf24;
        }

        .refresh-btn {
            background: #fbbf24;
            color: #111827;
            text-decoration: none;
            padding: 10px 18px;
            border-radius: 8px;
            font-weight: bold;
        }

        .container {
            max-width: 1200px;
            margin: 35px auto;
            padding: 0 20px;
        }

        .date {
            color: #666;
            margin-bottom: 25px;
        }

        .news-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 25px;
        }

        .card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        }

        .news-image {
            width: 100%;
            height: 260px;
            object-fit: cover;
        }

        .content {
            padding: 22px;
        }

        .content h2 {
            font-size: 23px;
            margin-bottom: 15px;
            line-height: 1.4;
        }

        .summary {
            color: #444;
            line-height: 1.8;
            white-space: pre-line;
        }

        .source {
            display: inline-block;
            margin-top: 20px;
            background: #111827;
            color: white;
            text-decoration: none;
            padding: 10px 16px;
            border-radius: 7px;
        }

        footer {
            text-align: center;
            padding: 25px;
            color: #777;
        }

        @media(max-width: 768px) {
            .news-grid {
                grid-template-columns: 1fr;
            }

            .header-content {
                gap: 15px;
                flex-direction: column;
            }
        }
    </style>
</head>

<body>

<header>
    <div class="header-content">
        <div class="logo">📰 AI <span>मराठी न्यूज</span></div>

        <a href="/" class="refresh-btn">🔄 Refresh News</a>
    </div>
</header>

<div class="container">

    <h1>आजच्या ताज्या बातम्या</h1>

    <p class="date">
        {{ current_date }}
    </p>

    <div class="news-grid">

        {% for item in news %}
        <div class="card">

            <img
                src="{{ item.image }}"
                class="news-image"
                alt="News Image"
            >

            <div class="content">

                <h2>{{ item.title }}</h2>

                <div class="summary">
                    {{ item.summary }}
                </div>

                <a
                    href="{{ item.link }}"
                    target="_blank"
                    class="source"
                >
                    मूळ बातमी वाचा →
                </a>

            </div>

        </div>
        {% endfor %}

    </div>

</div>

<footer>
    🤖 Powered by Ollama GPT-OSS 120B Cloud
</footer>

</body>
</html>
"""

# =========================================================
# GET NEWS FROM GOOGLE NEWS
# =========================================================

def get_news():
    feed = feedparser.parse(RSS_URL)
    news_list = []

    for item in feed.entries[:2]:

        english_title = item.title
        news_link = item.link

        # -------------------------------------------------
        # ASK GPT-OSS TO CREATE MARATHI NEWS
        # -------------------------------------------------

        prompt = f"""
तुम्ही मराठी न्यूज रिपोर्टर आहात.

आजची तारीख: {datetime.now().strftime('%d-%m-%Y')}

खालील बातमीचे शीर्षक आहे:

{english_title}

या बातमीसाठी मराठीत खालील माहिती तयार करा:

1. आकर्षक मराठी शीर्षक
2. 5 ते 6 ओळींचा सोपा आणि स्पष्ट बातमी सारांश
3. फक्त उपलब्ध शीर्षकावर आधारित माहिती द्या.
4. माहिती तयार करू नका किंवा खोटे तथ्य जोडू नका.

उत्तर खालील स्वरूपात द्या:

शीर्षक: [मराठी शीर्षक]

सारांश:
[बातमीचा सारांश]
"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180
            )

            response.raise_for_status()

            ai_response = response.json()["response"]

            # Separate title and summary
            marathi_title = english_title
            summary = ai_response

            if "शीर्षक:" in ai_response:
                parts = ai_response.split("सारांश:", 1)

                marathi_title = parts[0].replace(
                    "शीर्षक:", ""
                ).strip()

                if len(parts) > 1:
                    summary = parts[1].strip()

        except Exception as e:
            print("Ollama Error:", e)

            marathi_title = english_title
            summary = "या बातमीचा मराठी सारांश तयार करता आला नाही."

        # -------------------------------------------------
        # GENERATE AI IMAGE
        # -------------------------------------------------

        image_prompt = (
            f"Professional realistic editorial news photograph about "
            f"{english_title}, cinematic photography, high quality, "
            f"realistic, no text, no watermark"
        )

        image_url = (
            "https://image.pollinations.ai/prompt/"
            + quote(image_prompt)
            + "?width=800&height=500"
        )

        news_list.append({
            "title": marathi_title,
            "summary": summary,
            "link": news_link,
            "image": image_url
        })

    return news_list


# =========================================================
# FLASK HOME PAGE
# =========================================================

@app.route("/")
def home():

    news = get_news()

    return render_template_string(
        HTML,
        news=news,
        current_date=datetime.now().strftime("%d-%m-%Y")
    )


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    print("Starting AI Marathi News Website...")
    print("Open: http://127.0.0.1:5000")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )