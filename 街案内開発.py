import tkinter as tk  # GUI (グラフィカル・ユーザー・インターフェース) を構築するための標準ライブラリ
from tkinter import ttk  # tkinter のテーマ付きウィジェット (ボタンなど) を提供し、モダンな見た目にする
import requests  # HTTPリクエストを送信し、VOICEVOXやGemini APIと通信するためのライブラリ
import json  # JSON形式のデータをエンコード/デコードするための標準ライブラリ (API通信時に使用)
import sounddevice as sd  # 音声データを再生するためのライブラリ
import numpy as np  # 数値計算、特に音声データを数値配列として扱うために使用
import speech_recognition as sr  # マイクからの音声入力を処理し、テキストに変換するためのライブラリ
import time  # 時間関連の操作 (スリープなど) を行うための標準ライブラリ
import sys  # Pythonインタープリタとその環境に関する操作を行うための標準ライブラリ
import threading  # 時間のかかる処理をバックグラウンドで実行し、GUIの固まりを防ぐためのライブラリ
from PIL import Image, ImageTk  # 画像処理ライブラリ (Pillow)。画像の読み込み、リサイズ、Tkinterでの表示に使用
import cv2  # OpenCVライブラリ。動画ファイル (mp4など) の読み込みとフレーム処理に使用
import os  # オペレーティングシステムとのやり取り (ファイルのパス操作など) を行うための標準ライブラリ
import google.generativeai as genai  # Google Gemini API と通信するための公式SDK
import re  # 逐次再生のために re (正規表現) モジュールをインポート

# --- VOICEVOX & Gemini API 関連の設定 ---
VOICEVOX_HOST = "127.0.0.1"
VOICEVOX_PORT = "50021"
# 基本となるスピーカーID (例: 8: 四国めたんノーマル、1: ずんだもんノーマルなど、環境に合わせて変更してください)
VOICEVOX_SPEAKER_ID = 8 # 例: つむぎ (ここでは便宜的に8のままにします)

#追加項目
# VOICEVOXAPI公式での確認もしくは、卒論での参照の[1]をご覧ください。
# VOICEVOXのスタイルID設定 (使用するキャラクターに合わせて変更してください)
# ここでは、基本ID 8 のキャラクターが以下の感情スタイルIDに対応していることが前提。
# 実際の設定は、使用するVOICEVOXエンジンのキャラクターによって異なる。
# 例: つむぎ (デフォルト: 8)
#    喜びっぽいスタイル: 10 (例として適当なIDを設定。実際にはVOICEVOXで確認)
#    悲しみっぽいスタイル: 12 (例として適当なIDを設定。実際にはVOICEVOXで確認)
#    怒りっぽいスタイル: 14 (例として適当なIDを設定。実際にはVOICEVOXで確認)
#    驚きっぽいスタイル: 16 (例として適当なIDを設定。実際にはVOICEVOXで確認)
EMOTION_STYLE_MAP = {
    "ニュートラル": VOICEVOX_SPEAKER_ID,   # 基本スタイル
    "喜び": 10,       # 例: VOICEVOXで利用可能な「喜び」のスタイルID
    "悲しみ": 12,     # 例: VOICEVOXで利用可能な「悲しみ」のスタイルID
    "怒り": 14,        # 例: VOICEVOXで利用可能な「怒り」のスタイルID
    "驚き": 16,        # 例: VOICEVOXで利用可能な「驚き」のスタイルID
    # 対応するスタイルIDがない場合は、VOICEVOX_SPEAKER_ID が使用される。
}


# Gemini APIキーを設定してください。
GOOGLE_API_KEY = "AIzaSyDZ1OKt-MrEX7dgOScJ8GVC7EXK3IS40oM" # あなたのキーを設定
genai.configure(api_key=GOOGLE_API_KEY)

# --- 感情推論用関数 ---
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
        return "ニュートラル(普通)"

# --- 外部API通信用関数 (スタイルIDを追加) ---
def post_audio_query(text: str, style_id: int) -> dict | None:
    """VOICEVOXの音声合成クエリを作成する (スタイルIDを使用)"""
    params = {"text": text, "speaker": style_id}
    try:
        # タイムアウトを90秒に延長
        res = requests.post(f"http://{VOICEVOX_HOST}:{VOICEVOX_PORT}/audio_query", params=params, timeout=90)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print(f"\nVOICEVOX Audio Queryエラー (スタイルID: {style_id}): {e}")
        return None

def post_synthesis(query_data: dict, style_id: int) -> bytes | None:
    """VOICEVOXで音声合成を実行する (スタイルIDを使用)"""
    params = {"speaker": style_id}
    headers = {"content-type": "application/json"}
    try:
        # タイムアウトを90秒に延長（長い文章対策）
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
        print(f"\nVOICEVOX Synthesisエラー (スタイルID: {style_id}): {e}")
        return None

def play_wavfile(wav_data: bytes | None):
    """wavデータを再生する"""
    if wav_data is None:
        return
    try:
        sample_rate = 24000
        wav_array = np.frombuffer(wav_data, dtype=np.int16)
        sd.play(wav_array, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"\n音声再生エラー: {e}")

def recognize_speech_from_mic(recognizer: sr.Recognizer, microphone: sr.Microphone) -> dict:
    """マイクから音声を認識する"""
    response = {"success": True, "error": None, "transcription": None}
    with microphone as source:
        try:
            print("マイクのノイズレベルを調整中...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("どうぞ話してください（最大3秒間）...")
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
        except sr.WaitTimeoutError:
            response["success"] = False
            response["error"] = "タイムアウトしました。音声が検出されませんでした。"
            return response
        except Exception as e:
            response["success"] = False
            response["error"] = f"マイクからの音声取得中にエラー: {e}"
            return response

    try:
        response["transcription"] = recognizer.recognize_google(audio, language='ja-JP')
    except sr.RequestError as e:
        response["success"] = False
        response["error"] = f"Google APIに接続できませんでした; {e}"
    except sr.UnknownValueError:
        response["error"] = "音声を認識できませんでした"
    return response

def generate_gemini_response_with_emotion(prompt: str) -> tuple[str, str]:
    """
    Gemini APIを使用して応答を生成し、AIが発すべき推奨感情を返す
    return (応答テキスト, AIが発すべき推奨感情)
    """
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY":
        return "エラー: Gemini APIキーが設定されていません。", "ニュートラル"
        
    # ユーザーの感情を推論
    user_emotion = detect_user_emotion(prompt)

    # 観光地とキーワードのマッピング
    tourist_spots = {
        "渋谷": ["カフェ", "アクセス", "見どころ"],
        "浅草": ["食べ物", "雷門", "お土産"],
        "東京タワー": ["カフェ", "アクセス", "見どころ", "周辺", "周辺観光"]
    }

    # ユーザーの入力からキーワードを検出
    user_input_lower = prompt.lower()
    detected_spot = None
    detected_keyword = None

    for spot, keywords in tourist_spots.items():
        if spot in user_input_lower:
            detected_spot = spot
            for keyword in keywords:
                if keyword in user_input_lower:
                    detected_keyword = keyword
                    break
            break

    # 予測対話のプロンプトを生成
    system_prompt_base = "あなたは親切で感情豊かなAIアシスタントです。"
    
    if detected_spot and detected_keyword:
        # 観光地固有のプロンプト
        if detected_spot == "東京タワー":
            if "アクセス" in detected_keyword:
                system_prompt = "あなたは東京タワーの観光案内人です。ユーザーは東京タワーまでのアクセスについて知りたいようです。都営大江戸線「赤羽橋駅」や東京メトロ日比谷線「神谷町駅」など、主要な駅から東京タワーまでの行き方を分かりやすく、徒歩時間を含めて説明してください。"
            elif "カフェ" in detected_keyword:
                system_prompt = "あなたは東京タワーの観光案内人です。ユーザーは東京タワー周辺のおすすめカフェについて知りたいようです。東京タワーのメインデッキにある「カフェ ラ・トゥール」や、増上寺近くの「ル・パン・コティディアン 芝公園店」、麻布台ヒルズ内のカフェなど、具体的な店名を2〜3つ挙げて簡潔に紹介してください。"
            elif "見どころ" in detected_keyword or "周辺" in detected_keyword:
                system_prompt = "あなたは東京タワーの観光案内人です。ユーザーは東京タワー周辺の見どころや周辺観光について知りたいようです。増上寺、芝公園、麻布台ヒルズなど、徒歩で行ける主要なスポットを2〜3つ紹介してください。"
            else:
                system_prompt = f"あなたは{detected_spot}の観光案内人です。ユーザーの質問に親切に答えてください。"
        else:
            system_prompt = f"あなたは{detected_spot}の観光案内人です。ユーザーの質問に親切に答えてください。"
    else:
        system_prompt = system_prompt_base
    
    # ユーザーの感情に基づいたAIの応答トーンの指示
    emotion_instruction = ""
    ai_default_emotion = "ニュートラル" # AIが発すべきデフォルトの感情

    if user_emotion == "喜び":
        emotion_instruction = "ユーザーは今、とても喜んでいます。あなたの応答も、**喜びの感情**を込めたトーンで、共感し、さらに楽しい話題で会話を盛り上げるように応答してください。"
        ai_default_emotion = "喜び"
    elif user_emotion == "悲しみ":
        emotion_instruction = "ユーザーは今、悲しい気持ちです。あなたの応答は、**悲しみの感情**を込めたトーンで、共感し、優しい言葉をかけて慰めるように応答してください。"
        ai_default_emotion = "悲しみ"
    elif user_emotion == "怒り":
        emotion_instruction = "ユーザーは今、怒っています。あなたの応答は、**落ち着いたトーン**で、丁寧に状況を改善するように促す応答をしてください。"
        ai_default_emotion = "ニュートラル" # 怒りの感情で返すと対立するため、ニュートラルに抑える
    elif user_emotion == "驚き":
        emotion_instruction = "ユーザーは何か驚いたようです。あなたの応答は、**驚きや関心**を示すトーンで、その内容について質問し、詳細を引き出すように応答してください。"
        ai_default_emotion = "驚き"

    # 最終的なプロンプト
    final_prompt = f"{system_prompt} {emotion_instruction}\n\nユーザー: {prompt}"

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response_text = model.generate_content(final_prompt).text.strip()
        
        # 応答テキストからAIが発すべき感情を再推論（簡易的。Gemini自身の出力内容から判断）
        ai_emotion = ai_default_emotion # プロンプトで指定したデフォルト感情をベースにする
        if "！" in response_text or "すごい" in response_text or "わくわく" in response_text or "嬉しい" in response_text:
            ai_emotion = "喜び"
        elif "残念" in response_text or "お気の毒に" in response_text or "すみません" in response_text:
            ai_emotion = "悲しみ"
        elif "なんで" in response_text or "困りましたね" in response_text:
            ai_emotion = "怒り" # AIが困惑、少し不満のようなニュアンス
        elif "ええっ" in response_text or "まさか" in response_text:
            ai_emotion = "驚き"
            
        return response_text, ai_emotion
        
    except Exception as e:
        print(f"Gemini API呼び出し中にエラーが発生しました: {e}")
        return "Geminiとの通信中にエラーが発生しました。申し訳ありません。", "ニュートラル"

# --- GUIアプリケーションクラス ---
class VoiceChatApp:
    def __init__(self, master):
        self.master = master
        master.title("音声チャット")
        master.geometry("950x1080")
        
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        #スライドショーのサイズをここで定義（変更可能）
        self.slideshow_display_width = 400
        self.slideshow_display_height = 350
        
        # VRoid画像の表示サイズを調整
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
        
        # メディアファイルをここで読み込む
        self.load_media_files()
        
        self.initialize_microphone()
        
        # ウィンドウがマップされた後にon_resizeを呼び出す
        self.master.update_idletasks() # ウィンドウが完全に描画されるのを待つ
        self.on_resize(None) # 初期サイズでの設定
        self.master.bind("<Configure>", self.on_resize)
        
    def create_widgets(self):
        """GUIの各ウィジェットを作成・配置する"""
        # 背景画像
        self.bg_label = tk.Label(self.master)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.bg_label.lower()
        self.bg_image_original = None

        # VRoidキャラクター表示用ラベル
        self.vroid_label = tk.Label(self.master)
        self.vroid_label.place_forget() # 起動時は非表示
        
        # チャットログ用のフレーム
        chat_frame = tk.Frame(self.master, bg="SystemButtonFace")
        chat_frame.place(relx=0.5, rely=0.6, relwidth=0.8, relheight=0.3, anchor=tk.N) 
        
        # チャットログをフレーム内に配置
        self.chat_log = tk.Text(chat_frame, state=tk.DISABLED, font=("Meiryo", 12))
        self.chat_log.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # コントロールフレーム (入力欄と送信ボタン)
        control_frame = ttk.Frame(self.master)
        control_frame.place(relx=0.5, rely=0.85, relwidth=0.8, anchor=tk.N)
        
        self.input_entry = ttk.Entry(control_frame, font=("Meiryo", 12))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", self.send_message)
        
        self.send_button = ttk.Button(control_frame, text="送信", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5)

        button_frame = ttk.Frame(self.master)
        button_frame.place(relx=0.5, rely=0.9, anchor=tk.N)
        
        self.start_button = ttk.Button(button_frame, text="会話開始", command=self.start_conversation)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="会話停止", command=self.stop_conversation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.force_stop_button = ttk.Button(button_frame, text="強制終了", command=self.force_stop_conversation, state=tk.DISABLED)
        self.force_stop_button.pack(side=tk.LEFT, padx=5)
        
        slideshow_button_frame = ttk.Frame(self.master)
        slideshow_button_frame.place(relx=0.5, rely=0.95, anchor=tk.N)
        
        self.start_slideshow_button = ttk.Button(slideshow_button_frame, text="動画再生開始", command=self.start_video_slideshow)
        self.start_slideshow_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_slideshow_button = ttk.Button(slideshow_button_frame, text="動画再生停止", command=self.stop_video_slideshow, state=tk.DISABLED)
        self.stop_slideshow_button.pack(side=tk.LEFT, padx=5)
        
        self.next_slide_button = ttk.Button(slideshow_button_frame, text="次の動画", command=self.next_video, state=tk.DISABLED)
        self.next_slide_button.pack(side=tk.LEFT, padx=5)
        
        self.video_slideshow_label = None

    def load_media_files(self):
        """画像・動画ファイルを読み込む"""
        try:
            bg_image_path = os.path.join(self.base_path, "frame.jpg")
            self.bg_image_original = Image.open(bg_image_path)
        except FileNotFoundError:
            self.update_chat_log(f"エラー: 'frame.jpg' が見つかりません。", "red")
            self.bg_image_original = None

        try:
            vroid_char_path = os.path.join(self.base_path, "女の子.png")
            self.vroid_image_original = Image.open(vroid_char_path)
        except FileNotFoundError:
            self.update_chat_log(f"エラー: 'smalle.png' が見つかりません。", "red")
            self.vroid_image_original = None

        self.speaking_video_path = os.path.join(self.base_path, "video1.mp4")
        if not os.path.exists(self.speaking_video_path):
            self.update_chat_log(f"エラー: 'video1.mp4' が見つかりません。", "red")
        
        videos_folder_path = os.path.join(self.base_path, "videos")
        if os.path.isdir(videos_folder_path):
            self.video_files = [os.path.join(videos_folder_path, f) for f in os.listdir(videos_folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            self.video_files.sort()
            if not self.video_files:
                self.update_chat_log(f"警告: 'videos' フォルダに動画ファイルが見つかりません。", "orange")
        else:
            self.update_chat_log(f"警告: 'videos' フォルダが見つかりません。作成してください。", "orange")

    def initialize_microphone(self):
        """マイクの初期化を試みる"""
        try:
            self.microphone = sr.Microphone()
            self.update_chat_log("マイクの準備ができました。", "green")
            self.start_button.config(state=tk.NORMAL)
        except Exception as e:
            self.update_chat_log(f"エラー: マイクの初期化に失敗しました: {e}", "red")
            self.microphone = None
            self.start_button.config(state=tk.DISABLED)

    # --- アニメーション・メディア表示関連 ---
    def _start_speaking_animation(self):
        """話すアニメーションを開始する"""
        if not self.master.winfo_exists(): return
        if self.is_speaking_animation_active or not os.path.exists(self.speaking_video_path):
            return
        
        if self.is_video_slideshow_playing:
            self.stop_video_slideshow()
            
        self.is_speaking_animation_active = True
        self.speaking_cap = cv2.VideoCapture(self.speaking_video_path)
        if not self.speaking_cap.isOpened():
            self.update_chat_log(f"エラー: アニメーション動画を開けませんでした。", "red")
            self._end_speaking_animation()
            return
            
        # vroid_labelを表示
        self.vroid_label.place(relx=0.45, rely=0.1, anchor=tk.N)
        
        fps = self.speaking_cap.get(cv2.CAP_PROP_FPS)
        self.video_frame_delay = int(1000 / fps) if fps > 0 else 30
        self._play_speaking_animation_video()

    def _play_speaking_animation_video(self):
        """話すアニメーションの動画フレームを更新する"""
        if not self.master.winfo_exists(): return
        
        if self.is_speaking_animation_active and self.speaking_cap and self.speaking_cap.isOpened():
            ret, frame = self.speaking_cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # smalle.pngの元のサイズ（または指定したvroid_target_width/height）に固定してリサイズ
                resized_image = pil_image.resize((self.vroid_target_width, self.vroid_target_height), Image.Resampling.LANCZOS)
                self.speaking_vroid_photo = ImageTk.PhotoImage(resized_image)
                self.vroid_label.config(image=self.speaking_vroid_photo)
                self.vroid_label.image = self.speaking_vroid_photo
            else:
                self.speaking_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 動画をループ再生する
                
            self.master.after(self.video_frame_delay, self._play_speaking_animation_video)
        else:
            if self.speaking_cap:
                self.speaking_cap.release()
                self.speaking_cap = None

    def _end_speaking_animation(self):
        """話すアニメーションを終了し、通常画像に戻す"""
        if not self.master.winfo_exists(): return
        
        if self.is_speaking_animation_active:
            self.is_speaking_animation_active = False
            if self.speaking_cap:
                self.speaking_cap.release()
                self.speaking_cap = None
            
            # 会話が継続している場合のみ静止画に戻す
            if self.is_talking:
                self.resize_vroid_image()

    def resize_vroid_image(self):
        """通常のVRoid画像をリサイズして表示する"""
        if not self.master.winfo_exists(): return
        
        if self.vroid_image_original:
            # smalle.pngの元のサイズ（または指定したvroid_target_width/height）に固定してリサイズ
            resized_image = self.vroid_image_original.copy()
            resized_image.thumbnail((self.vroid_target_width, self.vroid_target_height), Image.Resampling.LANCZOS)

            self.vroid_photo = ImageTk.PhotoImage(resized_image)
            self.vroid_label.config(image=self.vroid_photo)
            self.vroid_label.image = self.vroid_photo
            self.vroid_label.place(relx=0.45, rely=0.1, anchor=tk.N)

    def on_resize(self, event):
        """ウィンドウサイズ変更時の処理"""
        if not self.master.winfo_exists(): return
        
        if self.bg_image_original:
            window_width = self.master.winfo_width()
            window_height = self.master.winfo_height()
            if window_width > 0 and window_height > 0:
                resized_bg = self.bg_image_original.resize((window_width, window_height), Image.Resampling.LANCZOS)
                self.bg_photo = ImageTk.PhotoImage(resized_bg)
                self.bg_label.config(image=self.bg_photo)
                self.bg_label.image = self.bg_photo
        
        if self.is_talking and not self.is_speaking_animation_active:
            self.resize_vroid_image()
        
    # --- スライドショー関連の関数 ---
    def start_video_slideshow(self):
        """動画スライドショーの再生を開始する"""
        if not self.master.winfo_exists(): return

        if not self.video_files:
            self.update_chat_log("動画スライドショーに表示する動画ファイルがありません。", "orange")
            return
            
        if self.is_video_slideshow_playing:
            return
        
        if self.is_speaking_animation_active:
            self.stop_conversation() # 会話中に動画スライドショーを再生しようとしたら会話を停止
            
        self.is_video_slideshow_playing = True
        
        # vroid_labelのサイズを動的に変更する
        self.vroid_label.config(width=self.slideshow_display_width, height=self.slideshow_display_height)
        self.vroid_label.place(relx=0.45, rely=0.1, anchor=tk.N)

        self.current_video_index = 0
        self.play_current_video()
        
        self.update_chat_log(f"動画スライドショーを開始します。", "green")
        self.start_slideshow_button.config(state=tk.DISABLED)
        self.stop_slideshow_button.config(state=tk.NORMAL)
        self.next_slide_button.config(state=tk.NORMAL)
        
    def stop_video_slideshow(self):
        """動画スライドショーの再生を停止する"""
        if not self.master.winfo_exists(): return

        if self.is_video_slideshow_playing:
            self.is_video_slideshow_playing = False
            if self.video_slideshow_after_id:
                self.master.after_cancel(self.video_slideshow_after_id)
                self.video_slideshow_after_id = None
            if self.current_video_cap:
                self.current_video_cap.release()
                self.current_video_cap = None
            
            if self.is_talking:
                self.resize_vroid_image()
            else:
                self.vroid_label.place_forget()

            self.update_chat_log("動画スライドショーを停止しました。", "orange")
            self.start_slideshow_button.config(state=tk.NORMAL)
            self.stop_slideshow_button.config(state=tk.DISABLED)
            self.next_slide_button.config(state=tk.DISABLED)

    def next_video(self):
        """次の動画に切り替える"""
        if not self.master.winfo_exists(): return
        if not self.is_video_slideshow_playing or not self.video_files:
            return
        
        if self.video_slideshow_after_id:
            self.master.after_cancel(self.video_slideshow_after_id)
            self.video_slideshow_after_id = None
        if self.current_video_cap:
            self.current_video_cap.release()
            self.current_video_cap = None
            
        self.current_video_index = (self.current_video_index + 1) % len(self.video_files)
        self.play_current_video()
        self.update_chat_log(f"次の動画に切り替えます: {os.path.basename(self.video_files[self.current_video_index])}", "green")

    def play_current_video(self):
        """現在のインデックスの動画を再生する"""
        if not self.master.winfo_exists(): return
        if not self.video_files:
            return
            
        self.current_video_cap = cv2.VideoCapture(self.video_files[self.current_video_index])
        if not self.current_video_cap.isOpened():
            self.update_chat_log(f"エラー: 動画ファイル '{os.path.basename(self.video_files[self.current_video_index])}' を開けませんでした。", "red")
            self.stop_video_slideshow()
            return
        
        fps = self.current_video_cap.get(cv2.CAP_PROP_FPS)
        self.video_frame_delay = int(1000 / fps) if fps > 0 else 30
        self._update_slideshow_frame()

    def _update_slideshow_frame(self):
        """スライドショーの動画フレームを更新する"""
        if not self.master.winfo_exists(): return
        
        if not self.is_video_slideshow_playing or not self.current_video_cap:
            return

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

    # --- 会話・音声認識・応答生成 ---
    def send_message(self, event=None):
        """テキスト入力からメッセージを送信し、応答を生成する"""
        if not self.master.winfo_exists(): return
        message = self.input_entry.get()
        if message:
            self.update_chat_log(f"あなた: {message}")
            self.input_entry.delete(0, tk.END)
            # handle_input は時間がかかるためスレッドで実行
            threading.Thread(target=self.handle_input, args=(message,)).start()

    # 逐次再生関数の修正
    def speak_sequentially(self, full_text: str, ai_emotion: str, on_finish=None):
        """テキストを文ごとに分割し、逐次合成・再生する (感情対応)"""
        if not self.master.winfo_exists(): return
        
        # AIの感情に基づいてスタイルIDを取得。見つからない場合は基本IDを使用
        style_id = EMOTION_STYLE_MAP.get(ai_emotion, VOICEVOX_SPEAKER_ID) 
        
        # 句読点（。！？）で分割し、句読点を残す正規表現
        sentences = re.findall(r'[^。！？\n]+[。！？\n]*', full_text.strip())

        def run_sequential_speak_async():
            #メインスレッドでアニメーションを開始する
            self.master.after(0, self._start_speaking_animation)
            
            try:
                for sentence in sentences:
                    if not self.is_talking:
                        # 会話停止ボタンが押されたら音声再生を中断
                        sd.stop() 
                        break
                        
                    # 1文ごとにVOICEVOX通信と再生を実行
                    # style_idを渡す
                    query_data = post_audio_query(sentence.strip(), style_id)
                    if query_data:
                        # style_idを渡す
                        wav_data = post_synthesis(query_data, style_id)
                        play_wavfile(wav_data)  
                        
            except Exception as e:
                print(f"逐次音声処理中に予期せぬエラー: {e}")
            finally:
                # 全文再生または中断後、アニメーションを終了し、後処理を実行
                self.master.after(0, self._end_speaking_animation)
                if on_finish and self.is_talking: # 会話が継続している場合のみon_finishを実行
                    self.master.after(0, on_finish)


        # 逐次再生処理全体を新しいスレッドで実行
        threading.Thread(target=run_sequential_speak_async).start()

    # 【重要】handle_input 関数の修正
    def handle_input(self, text):
        """入力テキストを処理し、応答を生成・再生する"""
        if not self.master.winfo_exists(): return
        
        # 応答生成は時間がかかるため、このスレッド（サブスレッド）で実行
        if "さようなら" in text or "バイバイ" in text:
            response_text = "はい、さようなら。またお話ししましょう。"
            ai_emotion = "ニュートラル" # 終了時の感情
            on_finish_action = self.stop_conversation
        else:
            # 修正: 感情も一緒に取得する新しい関数を呼び出す
            response_text, ai_emotion = generate_gemini_response_with_emotion(text)
            on_finish_action = None

            if not response_text:
                response_text = "申し訳ありません、応答を生成できませんでした。"
                ai_emotion = "ニュートラル" # エラー時はニュートラル
        
        # ログ表示はメインスレッドで実行
        self.master.after(0, self.update_chat_log, f"AI ({ai_emotion}): {response_text}", "blue")
        
        # 修正: 逐次再生を行う関数にAIの感情も渡す
        self.speak_sequentially(response_text, ai_emotion, on_finish=on_finish_action)

    def start_conversation(self):
        """会話スレッドを開始する"""
        if not self.master.winfo_exists(): return
        
        if self.microphone is None:
            self.update_chat_log("エラー: マイクが使用できません。", "red")
            return
        
        if not self.is_talking:
            self.is_talking = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.force_stop_button.config(state=tk.NORMAL)
            self.update_chat_log("会話を開始します。話しかけてください。", "green")

            self.resize_vroid_image()
            
            self.conversation_thread = threading.Thread(target=self.conversation_loop)
            self.conversation_thread.daemon = True
            self.conversation_thread.start()

    def conversation_loop(self):
        """音声認識ループ"""
        while self.is_talking:
            if not self.master.winfo_exists(): break

            self.update_chat_log("-" * 20)
            # 音声認識中はメインスレッドがブロックされない
            speech_response = recognize_speech_from_mic(self.recognizer, self.microphone)
            
            user_input = speech_response["transcription"]
            if self.is_talking and speech_response["success"] and user_input: # 会話停止されていないか再確認
                self.master.after(0, self.update_chat_log, f"あなた (音声): 「{user_input}」")
                # handle_input で応答生成・音声再生の全てが非同期で実行される
                threading.Thread(target=self.handle_input, args=(user_input,)).start()
            elif not speech_response["success"] and speech_response["error"] != "タイムアウトしました。音声が検出されませんでした。":
                self.master.after(0, self.update_chat_log, f"音声認識エラー: {speech_response['error']}", "red")
                # エラー応答も逐次再生（感情はニュートラルで）
                self.speak_sequentially("すみません、音声の認識で問題がありました。", "ニュートラル") 
            
            time.sleep(1)  

    def stop_conversation(self):
        """会話を停止する"""
        if not self.master.winfo_exists(): return
        
        if self.is_talking:
            self.is_talking = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.force_stop_button.config(state=tk.DISABLED)
            self.update_chat_log("会話を終了します。", "red")
            self._end_speaking_animation()
            
            self.vroid_label.place_forget()

    def force_stop_conversation(self):
        """会話を強制終了する"""
        if not self.master.winfo_exists(): return
        
        if self.is_talking:
            self.is_talking = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.force_stop_button.config(state=tk.DISABLED)
            self.update_chat_log("会話を強制終了します。", "purple")
            self._end_speaking_animation()
            
            self.vroid_label.place_forget()

    def update_chat_log(self, message, color="black"):
        """チャットログを更新する"""
        if not self.master.winfo_exists(): return
        
        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.insert(tk.END, message + "\n", color)
        self.chat_log.config(state=tk.DISABLED)
        self.chat_log.see(tk.END)
        self.chat_log.tag_config("red", foreground="red")
        self.chat_log.tag_config("blue", foreground="blue")
        self.chat_log.tag_config("green", foreground="green")
        self.chat_log.tag_config("purple", foreground="purple")
        self.chat_log.tag_config("orange", foreground="orange")
        
    def close_window(self):
        """ウィンドウを閉じる"""
        self.stop_conversation()
        self.stop_video_slideshow()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceChatApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_window)
    root.mainloop()
