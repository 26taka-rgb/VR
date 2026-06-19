
#© 2025 Yusuke

import tkinter as tk  # GUI構築用
from tkinter import ttk  # モダンなウィジェット
import requests  # HTTP通信用
import json  # JSON操作用
import sounddevice as sd  # 音声再生用
import numpy as np  # 数値計算用
import speech_recognition as sr  # 音声認識用
import time  # 時間制御用
import sys  # システム操作用
import threading  # 並列処理用
from PIL import Image, ImageTk  # 画像処理用
import cv2  # 動画処理用
import os  # ファイルパス操作用
from dotenv import load_dotenv  # 追加：.env読み込み用
#import google.generativeai as genai  # Gemini API用
# 新（推奨される書き方）
from google import genai
import re  # 正規表現用

# --- VOICEVOX 設定 ---
# 第2引数を設定しておくと、.envに書かれていない場合の「デフォルト値」になる
VOICEVOX_HOST = os.getenv("VOICEVOX_HOST", "127.0.0.1")
VOICEVOX_PORT = os.getenv("VOICEVOX_PORT", "50021")
# --- キャラクター設定 ---
# 引用: ユーザー提供コード (複数キャラクター管理機能)
# 画像ファイル、動画ファイル、性格プロンプトを管理
CHARACTER_SETTINGS = {
    "小日向由衣": {
        "speaker_id": 8,  # 春日部つむぎ（音声ID）
        "image_file": "女の子.png",
        "speaking_video": "video1.mp4",
        "personality_prompt": "あなたは「小日向由衣」です。東京都出身で、穏やかで人懐っこい、天然系癒しキャラの性格で、元気でフレンドリーな口調です。語尾に「だね」「だよ」などを使い、親しみやすく話してください。",
        "style_map": {
            "ニュートラル": 8, "喜び": 10, "悲しみ": 12, "怒り": 14, "驚き": 16
        }
    },
    "夜音美登": {
        "speaker_id": 8,
        "image_file": "夜音.png",
        "speaking_video": "video2.mp4",
        "personality_prompt": "あなたは「夜音美登」です。性格は一人でいると落ち着きがなく、甘えん坊です。語尾は「〜ね。」「〜だ。」「～よ。」「～だろう。」「～から。」を時々の場面中でこの語尾を入れて話してください。",
        "style_map": {
            "ニュートラル": 8, "喜び": 10, "悲しみ": 12, "怒り": 14, "驚き": 16
        }
    },
    "月橋希空": {
        "speaker_id": 8,
        "image_file": "月橋.png",
        "speaking_video": "video3.mp4",
        "personality_prompt": "あなたは「月橋希空」です。くだけた口調（よろしくね、じゃあ、またね！）で話してください。普段はおとなしく、趣味に対しては情熱的な性格です。",
        "style_map": {
            "ニュートラル": 8, "喜び": 10, "悲しみ": 12, "怒り": 14, "驚き": 16
        }
    }
}

# デフォルトのキャラクター名
DEFAULT_CHARACTER = "小日向由衣"

# .envファイルを読み込む
load_dotenv()

# 環境変数からAPIキーを取得
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# クライアントをグローバルで定義しておきます
client = None
if GOOGLE_API_KEY:
    client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    print("エラー: .env ファイルに GOOGLE_API_KEY が設定されていません。")

# --- 感情推論用関数 ---
# 引用: 街案内開発.py / ユーザー提供コード共通
def detect_user_emotion(text: str) -> str:
    """ユーザーの入力テキストから単純な感情を推論する"""
    text = text.lower()
    if "嬉しい" in text or "楽しい" in text or "よかった" in text or "最高" in text or "ありがとう" in text:
        return "喜び"
    elif "悲しい" in text or "辛い" in text or "残念" in text or "ひどい" in text or "つらい" in text:
        return "悲しみ"
    elif "怒り" in text or "ムカつく" in text or "許せない" in text or "なぜだ" in text or "最低" in text:
        return "怒り"
    elif "驚いた" in text or "マジで" in text or "信じられない" in text or "まさか" in text or "ええと" in text:
        return "驚き"
    else:
        return "ニュートラル"

# --- 外部API通信用関数 ---
# 引用: ユーザー提供コード (VoicevoxAPIと卒論の参照文献[1]一部含む引用)
def post_audio_query(text: str, style_id: int) -> dict | None:
    """VOICEVOXの音声合成クエリを作成する"""
    params = {"text": text, "speaker": style_id}
    try:
        res = requests.post(f"http://{VOICEVOX_HOST}:{VOICEVOX_PORT}/audio_query", params=params, timeout=90)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"\nVOICEVOX Audio Queryエラー (ID: {style_id}): {e}")
        return None

def post_synthesis(query_data: dict, style_id: int) -> bytes | None:
    """VOICEVOXで音声合成を実行する"""
    params = {"speaker": style_id}
    headers = {"content-type": "application/json"}
    try:
        res = requests.post(
            f"http://{VOICEVOX_HOST}:{VOICEVOX_PORT}/synthesis",
            data=json.dumps(query_data),
            params=params,
            headers=headers,
            timeout=90
        )
        res.raise_for_status()
        return res.content
    except requests.exceptions.RequestException as e:
        print(f"\nVOICEVOX Synthesisエラー (ID: {style_id}): {e}")
        return None

def play_wavfile(wav_data: bytes | None):
    """wavデータを再生する"""
    if wav_data is None:
        return
    try:
        # VOICEVOX標準のサンプリングレートに合わせて設定
        sample_rate = 24000
        wav_array = np.frombuffer(wav_data, dtype=np.int16)
        sd.play(wav_array, sample_rate)
        sd.wait() # 再生が終わるまで待機する
    except Exception as e:
        print(f"\n音声再生エラー: {e}")

def recognize_speech_from_mic(recognizer: sr.Recognizer, microphone: sr.Microphone) -> dict:
    """マイクから音声を認識する"""
    response = {"success": True, "error": None, "transcription": None}
    
    with microphone as source:
        try:
            print("マイクのノイズレベルを調整中...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("どうぞ話してください（最大3秒間）...")
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            response["success"] = False
            response["error"] = "タイムアウトしました。"
            return response
        except Exception as e:
            response["success"] = False
            response["error"] = f"マイクエラー: {e}"
            return response

    try:
        response["transcription"] = recognizer.recognize_google(audio, language='ja-JP')
    except sr.RequestError as e:
        response["success"] = False
        response["error"] = f"Google API接続エラー: {e}"
    except sr.UnknownValueError:
        response["error"] = "音声を認識できませんでした"
    return response

# --- GUIアプリケーションクラス ---
class VoiceChatApp:
    def __init__(self, master):
        self.master = master
        master.title("音声チャット - 街案内開発統合版")
        master.geometry("950x1080")
        
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        # --- 現在のキャラクター状態 ---
        self.current_char_name = DEFAULT_CHARACTER
        self.current_settings = CHARACTER_SETTINGS[DEFAULT_CHARACTER]

        # サイズ設定
        self.slideshow_display_width = 400
        self.slideshow_display_height = 350
        self.vroid_target_width = 350
        self.vroid_target_height = 350

        self.create_widgets()
        
        # --- 状態管理変数 ---
        self.vroid_image_original = None
        self.cap = None
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_talking = False
        self.conversation_thread = None
        self.is_speaking_animation_active = False
        self.speaking_cap = None
        self.speaking_vroid_photo = None

        self.video_files = []
        self.current_video_index = 0
        self.current_video_cap = None
        self.is_video_slideshow_playing = False
        self.video_slideshow_after_id = None
        self.speaking_video_path = None 

        # 初期ロード
        self.load_background()
        self.load_character_image() 
        self.resize_vroid_image()
        
        # 読み込み順序とロジックの整理
        self.load_videos() # スライドショー用
        self.load_speaking_video() # 話すときの動画用
        
        self.initialize_microphone()
        
        self.master.update_idletasks()
        self.on_resize(None)
        self.master.bind("<Configure>", self.on_resize)
        
    def create_widgets(self):
        """GUIの各ウィジェットを作成・配置する"""
        # 背景
        self.bg_label = tk.Label(self.master)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.bg_label.lower()
        self.bg_image_original = None

        # --- キャラクター選択エリア (上部) ---
        select_frame = tk.Frame(self.master, bg="#dddddd", bd=2, relief=tk.RAISED)
        select_frame.place(relx=0.5, rely=0.02, relwidth=0.5, height=40, anchor=tk.N)

        tk.Label(select_frame, text="キャラクター選択:", bg="#dddddd", font=("Meiryo", 10)).pack(side=tk.LEFT, padx=10)
        
        # コンボボックスの作成
        self.char_combo = ttk.Combobox(select_frame, values=list(CHARACTER_SETTINGS.keys()), font=("Meiryo", 10), state="readonly",width=20)
        self.char_combo.set(self.current_char_name) # 初期値設定
        self.char_combo.pack(side=tk.LEFT, padx=10)
        self.char_combo.bind("<<ComboboxSelected>>", self.change_character) # 変更時のイベントバインド

        # キャラクター表示用ラベル
        self.vroid_label = tk.Label(self.master)
        
        # --- チャットログ (スクロールバーあり) ---
        chat_frame = tk.Frame(self.master, bg="SystemButtonFace")
        chat_frame.place(relx=0.5, rely=0.6, relwidth=0.8, relheight=0.3, anchor=tk.N)  
        
        self.chat_log_scroll = ttk.Scrollbar(chat_frame)
        self.chat_log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.chat_log = tk.Text(chat_frame, state=tk.DISABLED, font=("Meiryo", 12),
                                yscrollcommand=self.chat_log_scroll.set)
        self.chat_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.chat_log_scroll.config(command=self.chat_log.yview)

        # コントロール (入力欄)
        control_frame = ttk.Frame(self.master)
        control_frame.place(relx=0.5, rely=0.85, relwidth=0.8, anchor=tk.N)
        
        self.input_entry = ttk.Entry(control_frame, font=("Meiryo", 12))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", self.send_message)
        
        self.send_button = ttk.Button(control_frame, text="送信", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5)

        # ボタン群
        button_frame = ttk.Frame(self.master)
        button_frame.place(relx=0.5, rely=0.9, anchor=tk.N)
        
        self.start_button = ttk.Button(button_frame, text="会話開始", command=self.start_conversation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="会話停止", command=self.stop_conversation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.force_stop_button = ttk.Button(button_frame, text="強制終了", command=self.force_stop_conversation, state=tk.DISABLED)
        self.force_stop_button.pack(side=tk.LEFT, padx=5)
        
        # スライドショーボタン
        slideshow_button_frame = ttk.Frame(self.master)
        slideshow_button_frame.place(relx=0.5, rely=0.95, anchor=tk.N)
        
        self.start_slideshow_button = ttk.Button(slideshow_button_frame, text="動画再生開始", command=self.start_video_slideshow)
        self.start_slideshow_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_slideshow_button = ttk.Button(slideshow_button_frame, text="動画再生停止", command=self.stop_video_slideshow, state=tk.DISABLED)
        self.stop_slideshow_button.pack(side=tk.LEFT, padx=5)
        
        self.next_slide_button = ttk.Button(slideshow_button_frame, text="次の動画", command=self.next_video, state=tk.DISABLED)
        self.next_slide_button.pack(side=tk.LEFT, padx=5)

    def change_character(self, event):
        """プルダウンでキャラクターが変更されたときの処理"""
        selected_name = self.char_combo.get()
        if selected_name == self.current_char_name:
            return

        self.update_chat_log(f"システム: キャラクターを「{selected_name}」に変更しました。", "purple")
        
        # 設定の更新
        self.current_char_name = selected_name
        self.current_settings = CHARACTER_SETTINGS[selected_name]
        
        # 画像の再読み込みと表示
        self.load_character_image()
        # キャラ変更時に必ずそのキャラの動画パスを読み込み直す
        self.load_speaking_video()
        
        # 常に画像を更新して表示する
        self.resize_vroid_image()

    def load_background(self):
        try:
            bg_image_path = os.path.join(self.base_path, "frame.jpg")
            self.bg_image_original = Image.open(bg_image_path)
        except FileNotFoundError:
            self.update_chat_log(f"エラー: 'frame.jpg' が見つかりません。", "red")
            self.bg_image_original = None

    def load_character_image(self):
        """現在のキャラクター設定に基づいて画像を読み込む"""
        file_name = self.current_settings["image_file"]
        try:
            char_path = os.path.join(self.base_path, file_name)
            self.vroid_image_original = Image.open(char_path)
        except FileNotFoundError:
            self.update_chat_log(f"エラー: キャラクター画像 '{file_name}' が見つかりません。", "red")
            self.vroid_image_original = None

    def load_speaking_video(self):
        """現在のキャラクターの設定から、話すときの動画パスを設定する"""
        video_filename = self.current_settings.get("speaking_video", "")
        if video_filename:
             self.speaking_video_path = os.path.join(self.base_path, video_filename)
        else:
             self.speaking_video_path = None

    def load_videos(self):
        """スライドショー用の動画を読み込む（話すときの動画とは別）"""
        videos_folder_path = os.path.join(self.base_path, "videos")
        if os.path.isdir(videos_folder_path):
            self.video_files = [os.path.join(videos_folder_path, f) for f in os.listdir(videos_folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            self.video_files.sort()
        else:
            self.update_chat_log(f"警告: 'videos' フォルダが見つかりません。", "orange")

    def initialize_microphone(self):
        try:
            self.microphone = sr.Microphone()
            self.update_chat_log("マイクの準備ができました。", "green")
            self.start_button.config(state=tk.NORMAL)
        except Exception as e:
            self.update_chat_log(f"エラー: マイクの初期化に失敗しました: {e}", "red")
            self.microphone = None
            self.start_button.config(state=tk.DISABLED)

    # --- Gemini 応答生成 (街案内ロジック統合版) ---
    def generate_gemini_response(self, prompt: str) -> tuple[str, str]:
        """
        Gemini APIを使用して応答を生成する。
        元の「街案内開発.py」の詳細な観光案内ロジックを統合しています。
        """
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY":
            return "エラー: Gemini APIキーが設定されていません。", "ニュートラル"
            
        user_emotion = detect_user_emotion(prompt)
        
        # 現在選択されているキャラクターの基本性格
        char_persona = self.current_settings["personality_prompt"]

        # 引用: 街案内開発.py (観光地ロジック部分)
        # --- 統合された観光地ロジック ---
        tourist_spots = {
            "渋谷": ["カフェ", "アクセス", "見どころ"],
            "浅草": ["食べ物", "雷門", "お土産"],
            "東京タワー": ["カフェ", "アクセス", "見どころ", "周辺", "周辺観光"]
        }

        user_input_lower = prompt.lower()
        detected_spot = None
        detected_keyword = None

        # 観光地とキーワードのマッチング
        for spot, keywords in tourist_spots.items():
            if spot in user_input_lower:
                detected_spot = spot
                for keyword in keywords:
                    if keyword in user_input_lower:
                        detected_keyword = keyword
                        break
                break

        # ガイド用システムプロンプトの構築 (引用: 街案内開発.py の詳細分岐ロジック)
        guide_instruction = ""
        
        if detected_spot:
            # 具体的なスポットが見つかった場合
            if detected_spot == "東京タワー":
                if detected_keyword and "アクセス" in detected_keyword:
                    guide_instruction = "あなたは東京タワーの観光案内人です。都営大江戸線「赤羽橋駅」や東京メトロ日比谷線「神谷町駅」など、主要な駅から東京タワーまでの行き方を分かりやすく、徒歩時間を含めて説明してください。"
                elif detected_keyword and "カフェ" in detected_keyword:
                    guide_instruction = "あなたは東京タワーの観光案内人です。東京タワーのメインデッキにある「カフェ ラ・トゥール」や、増上寺近くの「ル・パン・コティディアン 芝公園店」、麻布台ヒルズ内のカフェなど、具体的な店名を2〜3つ挙げて簡潔に紹介してください。"
                elif detected_keyword and ("見どころ" in detected_keyword or "周辺" in detected_keyword):
                    guide_instruction = "あなたは東京タワーの観光案内人です。増上寺、芝公園、麻布台ヒルズなど、徒歩で行ける主要なスポットを2〜3つ紹介してください。"
                else:
                    guide_instruction = f"あなたは{detected_spot}の観光案内人です。ユーザーの質問に親切に答えてください。"
            else:
                guide_instruction = f"あなたは{detected_spot}の観光案内人です。ユーザーの質問に親切に答えてください。"
        else:
            # スポット指定がない場合の基本スタンス
            guide_instruction = "あなたは親切なAIアシスタントです。観光案内人としての知識も持っています。"

        # 感情に基づいたトーン指示 (引用: 街案内開発.pyと自作コード)
        emotion_instruction = ""
        ai_default_emotion = "ニュートラル"

        if user_emotion == "喜び":
            emotion_instruction = "ユーザーは今、とても喜んでいます。あなたの応答も、**喜びの感情**を込めたトーンで、共感する。さらに楽しい話題で会話を盛り上げるように応答してください。"
            ai_default_emotion = "喜び"
        elif user_emotion == "悲しみ":
            emotion_instruction = "ユーザーは今、悲しい気持ちです。あなたの応答は、**悲しみの感情**を込めたトーンで、共感し、優しい言葉をかけて慰めるように応答してください。"
            ai_default_emotion = "悲しみ"
        elif user_emotion == "怒り":
            emotion_instruction = "ユーザーは今、怒っています。あなたの応答は、**落ち着いたトーン**で、丁寧に状況を改善するように促す応答をしてください。"
            ai_default_emotion = "ニュートラル"#標準処理
        elif user_emotion == "驚き":
            emotion_instruction = "ユーザーは何か驚いたようです。あなたの応答は、**驚きや関心**を示すトーンで、その内容について質問し、詳細を引き出すように応答してください。"
            ai_default_emotion = "驚き"

        # 最終プロンプト結合: キャラクター性格(ユーザー提供・指示) + 観光ガイド指示(街案内開発.py) + 感情指示(変更なし)
        final_prompt = f"{char_persona}\n{guide_instruction}\n{emotion_instruction}\n\nユーザー: {prompt}"

# (12個のスペース)
        try:
            # ここを追加：最初にデフォルトの感情を入れておく
            ai_emotion = ai_default_emotion 
            
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=final_prompt
            )
            response_text = response.text
            
            # 応答内容からAIの感情を再判定
            if "！" in response_text or "すごい" in response_text or "わくわく" in response_text:
                ai_emotion = "喜び"
            elif "残念" in response_text or "すみません" in response_text:
                ai_emotion = "悲しみ"
                
            return response_text, ai_emotion

        except Exception as e:
            print(f"Gemini API Error: {e}")
            # もしエラーになっても、ここで ai_default_emotion を使えば安全
            return "通信エラーが発生しました。", ai_default_emotion
    # --- アニメーション表示処理 ---
    def _start_speaking_animation(self):
        """話すときのアニメーションを開始"""
        if not self.master.winfo_exists():
            return
        
        # 動画パスが設定されていない、またはファイルが存在しない場合はアニメーションしない　(選択)
        if not self.speaking_video_path or not os.path.exists(self.speaking_video_path):
            return

        if self.is_speaking_animation_active: return
        
        if self.is_video_slideshow_playing: self.stop_video_slideshow()
            
        self.is_speaking_animation_active = True
        self.speaking_cap = cv2.VideoCapture(self.speaking_video_path)
        
        # vroid_labelの表示
        self.vroid_label.place(relx=0.45, rely=0.1, anchor=tk.N)
        
        fps = self.speaking_cap.get(cv2.CAP_PROP_FPS)
        self.video_frame_delay = int(1000 / fps) if fps > 0 else 30
        self._play_speaking_animation_video()

    def _play_speaking_animation_video(self):
        if not self.master.winfo_exists(): return
        
        if self.is_speaking_animation_active and self.speaking_cap and self.speaking_cap.isOpened():
            ret, frame = self.speaking_cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                # 現在の設定サイズに合わせてリサイズ
                resized_image = pil_image.resize((self.vroid_target_width, self.vroid_target_height), Image.Resampling.LANCZOS)
                self.speaking_vroid_photo = ImageTk.PhotoImage(resized_image)
                self.vroid_label.config(image=self.speaking_vroid_photo)
                self.vroid_label.image = self.speaking_vroid_photo
            else:
                self.speaking_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            self.master.after(self.video_frame_delay, self._play_speaking_animation_video)
        else:
            if self.speaking_cap: self.speaking_cap.release()

    def _end_speaking_animation(self):
        if not self.master.winfo_exists(): return
        if self.is_speaking_animation_active:
            self.is_speaking_animation_active = False
            if self.speaking_cap:
                self.speaking_cap.release()
                self.speaking_cap = None
            # 会話が続いている場合は静止画に戻す
            if self.is_talking:
                self.resize_vroid_image()

    def resize_vroid_image(self):
        """現在のキャラクター画像をリサイズして表示"""
        if not self.master.winfo_exists(): return
        
        if self.vroid_image_original:
            resized_image = self.vroid_image_original.copy()
            resized_image.thumbnail((self.vroid_target_width, self.vroid_target_height), Image.Resampling.LANCZOS)
            self.vroid_photo = ImageTk.PhotoImage(resized_image)
            self.vroid_label.config(image=self.vroid_photo)
            self.vroid_label.image = self.vroid_photo
            self.vroid_label.place(relx=0.45, rely=0.1, anchor=tk.N)
        else:
            self.vroid_label.place_forget()

    def on_resize(self, event):
        if not self.master.winfo_exists(): return
        if self.bg_image_original:
            w = self.master.winfo_width()
            h = self.master.winfo_height()
            if w > 0 and h > 0:
                resized_bg = self.bg_image_original.resize((w, h), Image.Resampling.LANCZOS)
                self.bg_photo = ImageTk.PhotoImage(resized_bg)
                self.bg_label.config(image=self.bg_photo)
                self.bg_label.image = self.bg_photo
            
        self.resize_vroid_image()

    # -- スライドショー機能 --
    def start_video_slideshow(self):
        if not self.master.winfo_exists(): return
        if not self.video_files:
            self.update_chat_log("再生する動画がありません。", "orange")
            return
        if self.is_video_slideshow_playing: return
        if self.is_speaking_animation_active: self.stop_conversation()
            
        self.is_video_slideshow_playing = True
        self.vroid_label.config(width=self.slideshow_display_width, height=self.slideshow_display_height)
        self.vroid_label.place(relx=0.45, rely=0.1, anchor=tk.N)
        self.current_video_index = 0
        self.play_current_video()
        self.start_slideshow_button.config(state=tk.DISABLED)
        self.stop_slideshow_button.config(state=tk.NORMAL)
        self.next_slide_button.config(state=tk.NORMAL)

    def stop_video_slideshow(self):
        if not self.master.winfo_exists(): return
        if self.is_video_slideshow_playing:
            self.is_video_slideshow_playing = False
            if self.video_slideshow_after_id:
                self.master.after_cancel(self.video_slideshow_after_id)
                self.video_slideshow_after_id = None
            if self.current_video_cap:
                self.current_video_cap.release()
                self.current_video_cap = None
            
            self.resize_vroid_image()

            self.start_slideshow_button.config(state=tk.NORMAL)
            self.stop_slideshow_button.config(state=tk.DISABLED)
            self.next_slide_button.config(state=tk.DISABLED)

    def next_video(self):
        if not self.master.winfo_exists() or not self.is_video_slideshow_playing: return
        if self.video_slideshow_after_id: self.master.after_cancel(self.video_slideshow_after_id)
        if self.current_video_cap: self.current_video_cap.release()
        self.current_video_index = (self.current_video_index + 1) % len(self.video_files)
        self.play_current_video()

    def play_current_video(self):
        if not self.video_files: return
        self.current_video_cap = cv2.VideoCapture(self.video_files[self.current_video_index])
        fps = self.current_video_cap.get(cv2.CAP_PROP_FPS)
        self.video_frame_delay = int(1000 / fps) if fps > 0 else 30
        self._update_slideshow_frame()

    def _update_slideshow_frame(self):
        if not self.master.winfo_exists() or not self.is_video_slideshow_playing or not self.current_video_cap: return
        ret, frame = self.current_video_cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            resized_image = pil_image.resize((self.slideshow_display_width, self.slideshow_display_height), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(resized_image)
            self.vroid_label.config(image=tk_image)
            self.vroid_label.image = tk_image
            self.video_slideshow_after_id = self.master.after(self.video_frame_delay, self._update_slideshow_frame)
        else:
            self.next_video()

    # --- 会話ロジック ---  #マイクロソフト公式一部用
    def send_message(self, event=None):
        """テキスト入力によるメッセージ送信"""
        if not self.master.winfo_exists(): return
        message = self.input_entry.get()
        if message:
            self.update_chat_log(f"あなた: {message}")
            self.input_entry.delete(0, tk.END)
            
            # メッセージ送信後すぐに静止画に戻しておく（動画再生前の一時的な処置）
            self.master.after(0, self.resize_vroid_image)
            
            threading.Thread(target=self.handle_input, args=(message, False)).start()

    def speak_and_wait(self, full_text: str, ai_emotion: str):
        """VOICEVOXを同期的に実行し、再生が終わるのを待つ"""
        if not self.master.winfo_exists(): return
        
        style_map = self.current_settings.get("style_map", {})
        base_id = self.current_settings["speaker_id"]
        style_id = style_map.get(ai_emotion, base_id)
        
        sentences = re.findall(r'[^。！？\n]+[。！？\n]*', full_text.strip())

        self.master.after(0, self._start_speaking_animation)
        try:
            for sentence in sentences:
                if not self.is_talking:
                    sd.stop()
                    break
                query_data = post_audio_query(sentence.strip(), style_id)
                if query_data:
                    wav_data = post_synthesis(query_data, style_id)
                    play_wavfile(wav_data)
        except Exception as e:
            print(f"Speech Error: {e}")
        finally:
            self.master.after(0, self._end_speaking_animation)
            
    def speak_sequentially(self, full_text: str, ai_emotion: str, on_finish=None): #音声非同期でも可能です。
        """非同期で音声を生成・再生"""
        def run_sequential_speak_async():
            self.speak_and_wait(full_text, ai_emotion)
            if on_finish and self.is_talking:
                self.master.after(0, on_finish)

        threading.Thread(target=run_sequential_speak_async).start()

    def handle_input(self, text, is_voice_input=True):
        """ユーザー入力（テキストまたは音声）を処理する"""
        if not self.master.winfo_exists(): return
        
        if "さようなら" in text:
            response_text = "さようなら。またお話ししましょう。"
            ai_emotion = "ニュートラル"
            on_finish_action = self.stop_conversation
        else:
            response_text, ai_emotion = self.generate_gemini_response(text)
            on_finish_action = None

        self.master.after(0, self.update_chat_log, f"{self.current_char_name} ({ai_emotion}): {response_text}", "blue")

        if is_voice_input:
            self.speak_and_wait(response_text, ai_emotion)
            if on_finish_action:
                 self.master.after(0, on_finish_action)
        else:
            self.speak_sequentially(response_text, ai_emotion, on_finish=on_finish_action)

    def start_conversation(self):
        if not self.master.winfo_exists(): return
        if self.microphone is None:
            self.update_chat_log("エラー: マイク使用不可", "red")
            return
        
        if not self.is_talking:
            self.is_talking = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.force_stop_button.config(state=tk.NORMAL)
            self.update_chat_log(f"会話開始 ({self.current_char_name})", "green")
            
            self.resize_vroid_image()
            
            self.conversation_thread = threading.Thread(target=self.conversation_loop)
            self.conversation_thread.daemon = True
            self.conversation_thread.start()

    def conversation_loop(self):
        while self.is_talking:
            if not self.master.winfo_exists(): break
            self.update_chat_log("-" * 20)
            
            speech_response = recognize_speech_from_mic(self.recognizer, self.microphone)
            user_input = speech_response["transcription"]
            
            if self.is_talking and speech_response["success"] and user_input:
                self.master.after(0, self.update_chat_log, f"あなた (音声): 「{user_input}」")
                self.handle_input(user_input, is_voice_input=True) 
                
            elif not speech_response["success"] and speech_response["error"] != "タイムアウトしました。":
                self.master.after(0, self.update_chat_log, f"エラー: {speech_response['error']}", "red")

    def stop_conversation(self):
        if not self.master.winfo_exists(): return
        if self.is_talking:
            self.is_talking = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.force_stop_button.config(state=tk.DISABLED)
            self.update_chat_log("会話終了", "red")
            self._end_speaking_animation()
            self.resize_vroid_image() 

    def force_stop_conversation(self):
        self.stop_conversation()

    def update_chat_log(self, message, color="black"):
        if not self.master.winfo_exists(): return
        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.insert(tk.END, message + "\n", color)
        self.chat_log.config(state=tk.DISABLED)
        self.chat_log.see(tk.END)
        self.chat_log.tag_config("red", foreground="red")
        self.chat_log.tag_config("blue", foreground="grey")
        self.chat_log.tag_config("green", foreground="green")
        self.chat_log.tag_config("purple", foreground="purple")
        self.chat_log.tag_config("orange", foreground="blue")

    def close_window(self):
        self.stop_conversation()
        self.stop_video_slideshow()
        self.master.destroy()

#メイン関数
if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceChatApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_window)
    root.mainloop()
