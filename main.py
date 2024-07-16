import streamlit as st
import pandas as pd
from apify_client import ApifyClient
import os
from datetime import datetime, timedelta
import random
import anthropic
from prompt import system_prompt_insight_analysis

# Apifyの設定
APIFY_API_KEY = st.secrets["APIFY_API_KEY"]
GOOGLE_MAPS_SCRAPER_ID = 'compass/Google-Maps-Reviews-Scraper'

# ApifyClientの初期化
client = ApifyClient(APIFY_API_KEY)

# 出力ディレクトリとファイル名を指定
output_dir = 'output'
output_file_path = os.path.join(output_dir, 'reviews_data.csv')
output_file_path_ai = os.path.join(output_dir, 'reviews_data_ai.csv')

# 出力ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# Streamlitアプリの設定
st.set_page_config(layout="wide", page_title="Google Maps Reviews Analyzer")
st.title('Google Maps Reviews Analyzer')

# スクレイピング関数
def scrape_google_maps_reviews(url, max_reviews):
    run_input = {
        "language": "ja",
        "maxReviews": max_reviews,
        "personalData": True,
        "startUrls": [{"url": url}]
    }
    
    run = client.actor(GOOGLE_MAPS_SCRAPER_ID).call(run_input=run_input)
    
    reviews_data = []
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        reviews_data.append(item)
    
    return reviews_data

# データ処理関数
def process_reviews_data(reviews_data):
    df = pd.DataFrame(reviews_data)
    df['stars'] = pd.to_numeric(df['stars'])
    
    # レビュー日をランダムに生成（過去2年以内）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2年前
    df['review_date'] = [start_date + timedelta(days=random.randint(0, 730)) for _ in range(len(df))]
    df['review_date'] = pd.to_datetime(df['review_date'])
    
    return df

# グラフ作成関数
def create_graphs(df):
    col1, col2 = st.columns(2)
    
    with col1:
        # 星評価の分布
        st.subheader('星評価の分布')
        star_counts = df['stars'].value_counts().sort_index()
        st.bar_chart(star_counts)
    
    with col2:
        # 時系列でのレビュー数
        st.subheader('時間経過とレビュー数')
        df_time = df.set_index('review_date').resample('M').size().reset_index(name='count')
        st.line_chart(df_time.set_index('review_date'))

# AI分析用の関数
def get_ai_analysis(api_key, prompt, user_message):
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=3000,
        temperature=0.7,
        system=prompt,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    return message.content[0].text

# タブの作成
tab1, tab2 = st.tabs(["データ取得・分析", "プロンプト編集"])

with tab1:
    # 入力フィールド
    col1, col2 = st.columns(2)
    with col1:
        google_maps_url = st.text_input("Google Maps URLを入力してください")
    with col2:
        max_reviews = st.number_input("取得する口コミの最大数", min_value=1, max_value=1000, value=100)

    # メイン処理
    if st.button('口コミを取得する'):
        if google_maps_url:
            with st.spinner('口コミを取得中...'):
                reviews_data = scrape_google_maps_reviews(google_maps_url, max_reviews)
                df = process_reviews_data(reviews_data)
                
                # セッションステートにデータを保存
                st.session_state['df'] = df
                st.session_state['data_loaded'] = True

    # データが読み込まれている場合、表示する
    if 'data_loaded' in st.session_state and st.session_state['data_loaded']:
        df = st.session_state['df']
        
        # 基本情報の表示
        st.subheader('店舗基本情報')
        st.write(f"店舗名: {df['title'].iloc[0]}")
        st.write(f"カテゴリ: {df['categoryName'].iloc[0]}")
        st.write(f"総レビュー数: {df['reviewsCount'].iloc[0]}")
        st.write(f"平均評価: {df['stars'].mean():.2f}")
        
        # CSVに保存
        df.to_csv(output_file_path, index=False)
        
        # データフレームを表示
        st.subheader('レビューデータ')
        st.dataframe(df[['review_date', 'stars', 'text']])
        
        # グラフを表示
        create_graphs(df)
        
        # CSVダウンロードボタン
        st.download_button(
            label="CSVをダウンロード",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="reviews_data.csv",
            mime='text/csv'
        )
        
        # AI分析
        st.subheader("AI分析")
        if st.button("AI分析を実行"):
            # AI分析用にデータを絞り込む
            df_ai = df[['stars', 'text']].dropna(subset=['text'])
            df_ai.to_csv(output_file_path_ai, index=False)
            
            with open(output_file_path_ai, 'r') as f:
                csv_data = f.read()
            
            user_message = f"以下はGoogleマップのレビューデータです。このデータを分析してください。データには評価（stars）とレビューテキスト（text）が含まれています。\n\n{csv_data}"
            
            with st.spinner('AI分析を実行中...'):
                analysis_result = get_ai_analysis(st.secrets["ANTHROPIC_API_KEY"], system_prompt_insight_analysis, user_message)
                st.text_area("AI分析結果", analysis_result, height=400)
    else:
        st.info("Google Maps URLを入力し、「口コミを取得する」ボタンを押してデータを取得してください。")

with tab2:
    st.header("プロンプト編集")
    
    edited_prompt = st.text_area("システムプロンプト", system_prompt_insight_analysis, height=300)
    
    if st.button("プロンプトを保存"):
        # プロンプトをファイルに保存
        with open('prompt.py', 'w') as f:
            f.write(f"system_prompt_insight_analysis = '''{edited_prompt}'''")
        st.success("プロンプトが保存されました。")