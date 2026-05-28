import nltk
nltk.download('movie_reviews', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
import random
import re
import threading
import tkinter as tk
from tkinter import font as tkfont

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ─────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────
C = {
    'bg'      : '#0F0F13',
    'surface' : '#1A1A24',
    'border'  : '#2A2A3A',
    'pos'     : '#1D9E75',
    'pos_lt'  : '#E1F5EE',
    'neg'     : '#D85A30',
    'neg_lt'  : '#FAECE7',
    'blue'    : '#378ADD',
    'purple'  : '#7F77DD',
    'text'    : '#F0EEE8',
    'muted'   : '#888780',
    'accent'  : '#7F77DD',
}

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────
def load_and_train(status_cb):
    status_cb("Loading dataset…")
    documents = [
        (movie_reviews.raw(fileid), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]
    random.seed(42)
    random.shuffle(documents)
    texts  = [t for t, _ in documents]
    labels = [1 if l == "pos" else 0 for _, l in documents]

    stop_words = set(stopwords.words('english'))
    negations  = {"no","not","nor","never","neither","nobody","nothing","nowhere","hardly","barely"}
    stop_words -= negations
    lemmatizer = WordNetLemmatizer()
    tokenizer  = TreebankWordTokenizer()

    def preprocess(text):
        text   = text.lower()
        text   = re.sub(r'<.*?>', ' ', text)
        text   = re.sub(r'[^a-z\s]', ' ', text)
        tokens = tokenizer.tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
        return " ".join(tokens)

    status_cb("Preprocessing 2 000 reviews…")
    texts_clean = [preprocess(t) for t in texts]

    status_cb("Training model…")
    X_train, X_test, y_train, y_test = train_test_split(
        texts_clean, labels, test_size=0.2, stratify=labels, random_state=42
    )
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2),
                                 sublinear_tf=True, min_df=3)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    nb  = MultinomialNB(alpha=0.1)
    lr  = LogisticRegression(max_iter=1000, C=5, random_state=42)
    ens = VotingClassifier(estimators=[('nb', nb), ('lr', lr)], voting='soft')
    ens.fit(X_train_vec, y_train)

    y_pred  = ens.predict(X_test_vec)
    y_proba = ens.predict_proba(X_test_vec)[:, 1]
    acc                      = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    status_cb("Ready")
    return {
        'vectorizer'  : vectorizer,
        'ensemble'    : ens,
        'preprocess'  : preprocess,
        'acc'         : acc,
        'precision'   : precision,
        'recall'      : recall,
        'f1'          : f1,
        'y_test'      : y_test,
        'y_pred'      : y_pred,
        'y_proba'     : y_proba,
        'texts_clean' : texts_clean,
        'labels'      : labels,
    }

# ─────────────────────────────────────────────
# 2. GUI
# ─────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Movie Sentiment Analyzer")
        self.configure(bg=C['bg'])
        self.geometry("900x680")
        self.resizable(True, True)
        self.model_data = None
        self._build_ui()
        threading.Thread(target=self._train, daemon=True).start()

    # ── helpers ──────────────────────────────
    def _lbl(self, parent, text, size=12, weight='normal', color=None, **kw):
        return tk.Label(parent, text=text,
                        font=(tkfont.nametofont("TkDefaultFont").actual()['family'], size, weight),
                        fg=color or C['text'], bg=kw.pop('bg', C['bg']), **kw)

    def _frame(self, parent, bg=None, **kw):
        return tk.Frame(parent, bg=bg or C['bg'], **kw)

    # ── layout ───────────────────────────────
    def _build_ui(self):
        # ── header ──
        hdr = self._frame(self, bg=C['surface'])
        hdr.pack(fill='x')
        tk.Frame(hdr, bg=C['accent'], width=4).pack(side='left', fill='y')
        inner = self._frame(hdr, bg=C['surface'])
        inner.pack(side='left', padx=16, pady=12)
        self._lbl(inner, "🎬  Movie Sentiment Analyzer", 16, 'bold',
                  bg=C['surface']).pack(anchor='w')
        self._lbl(inner, "NLP · TF-IDF · Ensemble Classifier", 10,
                  color=C['muted'], bg=C['surface']).pack(anchor='w')

        self.status_var = tk.StringVar(value="Starting up…")
        self._lbl(hdr, "", 10, color=C['muted'], bg=C['surface'],
                  textvariable=self.status_var).pack(side='right', padx=16)

        # ── tabs ──
        tab_bar = self._frame(self, bg=C['surface'])
        tab_bar.pack(fill='x')
        self.tab_btns   = {}
        self.tab_frames = {}
        self.active_tab = tk.StringVar(value='predict')

        for name, label in [('predict','Analyze Review'), ('metrics','Model Stats'), ('charts','Charts')]:
            b = tk.Button(tab_bar, text=label, relief='flat', bd=0,
                          font=(tkfont.nametofont("TkDefaultFont").actual()['family'], 10),
                          fg=C['text'], bg=C['surface'], activebackground=C['bg'],
                          activeforeground=C['accent'], padx=16, pady=8,
                          command=lambda n=name: self._switch_tab(n))
            b.pack(side='left')
            self.tab_btns[name] = b

        sep = tk.Frame(self, bg=C['border'], height=1)
        sep.pack(fill='x')

        # ── content area ──
        content = self._frame(self)
        content.pack(fill='both', expand=True, padx=0, pady=0)

        self.tab_frames['predict'] = self._build_predict_tab(content)
        self.tab_frames['metrics'] = self._build_metrics_tab(content)
        self.tab_frames['charts']  = self._build_charts_tab(content)

        self._switch_tab('predict')

    def _switch_tab(self, name):
        for n, f in self.tab_frames.items():
            f.pack_forget()
        self.tab_frames[name].pack(fill='both', expand=True)
        for n, b in self.tab_btns.items():
            b.config(fg=C['accent'] if n == name else C['text'],
                     bg=C['bg'] if n == name else C['surface'])
        self.active_tab.set(name)

    # ── predict tab ──────────────────────────
    def _build_predict_tab(self, parent):
        f = self._frame(parent)

        top = self._frame(f)
        top.pack(fill='x', padx=24, pady=(20, 0))
        self._lbl(top, "Write your movie review", 13, 'bold').pack(anchor='w')
        self._lbl(top, "The model will classify it as positive or negative",
                  10, color=C['muted']).pack(anchor='w', pady=(2,0))

        box_f = self._frame(f, bg=C['surface'],
                            highlightbackground=C['border'],
                            highlightthickness=1)
        box_f.pack(fill='x', padx=24, pady=12)
        self.text_input = tk.Text(box_f, height=6, wrap='word',
                                  bg=C['surface'], fg=C['text'],
                                  insertbackground=C['text'],
                                  font=('Courier', 11),
                                  relief='flat', padx=12, pady=10,
                                  selectbackground=C['accent'])
        self.text_input.pack(fill='x')
        self.text_input.bind('<Control-Return>', lambda e: self._analyze())

        btn_row = self._frame(f)
        btn_row.pack(fill='x', padx=24)
        analyze_btn = tk.Button(btn_row, text="Analyze  ▶",
                                command=self._analyze,
                                bg=C['accent'], fg='white',
                                relief='flat', bd=0,
                                font=(tkfont.nametofont("TkDefaultFont").actual()['family'], 11, 'bold'),
                                padx=20, pady=8,
                                activebackground='#6B63C4',
                                cursor='hand2')
        analyze_btn.pack(side='left')
        clear_btn = tk.Button(btn_row, text="Clear",
                              command=self._clear,
                              bg=C['surface'], fg=C['muted'],
                              relief='flat', bd=0,
                              font=(tkfont.nametofont("TkDefaultFont").actual()['family'], 10),
                              padx=14, pady=8,
                              activebackground=C['border'],
                              cursor='hand2')
        clear_btn.pack(side='left', padx=8)
        self._lbl(btn_row, "Ctrl+Enter to analyze", 9,
                  color=C['muted']).pack(side='right')

        # result card
        self.result_card = self._frame(f, bg=C['surface'],
                                       highlightbackground=C['border'],
                                       highlightthickness=1)
        self.result_card.pack(fill='x', padx=24, pady=16)

        rc_inner = self._frame(self.result_card, bg=C['surface'])
        rc_inner.pack(fill='x', padx=20, pady=16)

        self.result_emoji = self._lbl(rc_inner, "🎬", 36, bg=C['surface'])
        self.result_emoji.pack()
        self.result_label = self._lbl(rc_inner, "Waiting for input…", 18, 'bold',
                                      color=C['muted'], bg=C['surface'])
        self.result_label.pack(pady=(4, 0))
        self.result_conf  = self._lbl(rc_inner, "", 11, color=C['muted'], bg=C['surface'])
        self.result_conf.pack()

        # bar
        bar_wrap = self._frame(rc_inner, bg=C['surface'])
        bar_wrap.pack(fill='x', pady=(12,0))
        self._lbl(bar_wrap, "Negative", 9, color=C['neg'], bg=C['surface']).pack(side='left')
        self._lbl(bar_wrap, "Positive", 9, color=C['pos'], bg=C['surface']).pack(side='right')
        track = self._frame(rc_inner, bg=C['border'], height=6)
        track.pack(fill='x', pady=(2,0))
        track.pack_propagate(False)
        self.conf_bar = self._frame(track, bg=C['muted'], height=6)
        self.conf_bar.place(x=0, y=0, relheight=1, relwidth=0.5)

        # history
        self._lbl(f, "History", 12, 'bold').pack(anchor='w', padx=24, pady=(8,4))
        hist_wrap = self._frame(f, bg=C['surface'],
                                highlightbackground=C['border'],
                                highlightthickness=1)
        hist_wrap.pack(fill='both', expand=True, padx=24, pady=(0,16))
        self.history_box = tk.Text(hist_wrap, state='disabled', wrap='word',
                                   bg=C['surface'], fg=C['text'],
                                   font=('Courier', 10), relief='flat',
                                   padx=10, pady=8, height=6)
        self.history_box.pack(fill='both', expand=True)
        self.history_box.tag_config('pos', foreground=C['pos'])
        self.history_box.tag_config('neg', foreground=C['neg'])
        self.history_box.tag_config('dim', foreground=C['muted'])

        return f

    def _analyze(self, _=None):
        if not self.model_data:
            return
        text = self.text_input.get('1.0', 'end').strip()
        if not text:
            return
        md        = self.model_data
        clean     = md['preprocess'](text)
        vec       = md['vectorizer'].transform([clean])
        pred      = md['ensemble'].predict(vec)[0]
        proba     = md['ensemble'].predict_proba(vec)[0]
        pos_prob  = proba[1]
        conf      = max(proba) * 100
        is_pos    = pred == 1

        self.result_emoji.config(text="😊" if is_pos else "😞")
        self.result_label.config(
            text="POSITIVE" if is_pos else "NEGATIVE",
            fg=C['pos'] if is_pos else C['neg']
        )
        self.result_conf.config(text=f"Confidence: {conf:.1f}%")
        self.conf_bar.config(bg=C['pos'] if is_pos else C['neg'])
        self.conf_bar.place(relwidth=pos_prob)

        # add to history
        short = text[:60] + ("…" if len(text) > 60 else "")
        tag   = 'pos' if is_pos else 'neg'
        icon  = "✔" if is_pos else "✘"
        self.history_box.config(state='normal')
        self.history_box.insert('1.0', f'\n', 'dim')
        self.history_box.insert('1.0', f'  "{short}"\n', 'dim')
        self.history_box.insert('1.0', f'{icon} {"POSITIVE" if is_pos else "NEGATIVE"} ({conf:.0f}%)  ', tag)
        self.history_box.config(state='disabled')

    def _clear(self):
        self.text_input.delete('1.0', 'end')
        self.result_emoji.config(text="🎬")
        self.result_label.config(text="Waiting for input…", fg=C['muted'])
        self.result_conf.config(text="")
        self.conf_bar.config(bg=C['muted'])
        self.conf_bar.place(relwidth=0.5)

    # ── metrics tab ──────────────────────────
    def _build_metrics_tab(self, parent):
        f = self._frame(parent)
        self.metrics_inner = self._frame(f)
        self.metrics_inner.pack(fill='both', expand=True, padx=24, pady=20)
        self._lbl(self.metrics_inner, "Training model, please wait…",
                  13, color=C['muted']).pack(pady=40)
        return f

    def _populate_metrics(self):
        for w in self.metrics_inner.winfo_children():
            w.destroy()
        md = self.model_data
        self._lbl(self.metrics_inner, "Model Performance", 14, 'bold').pack(anchor='w', pady=(0,12))

        grid = self._frame(self.metrics_inner)
        grid.pack(fill='x')
        cards = [
            ("Accuracy",  f"{md['acc']*100:.1f}%",  "overall correct predictions"),
            ("Precision", f"{md['precision']*100:.1f}%", "of predicted positives, truly positive"),
            ("Recall",    f"{md['recall']*100:.1f}%",    "of actual positives, correctly found"),
            ("F1-score",  f"{md['f1']*100:.1f}%",        "harmonic mean of precision & recall"),
        ]
        colors = [C['blue'], C['pos'], C['purple'], C['neg']]
        for i, ((label, val, desc), col) in enumerate(zip(cards, colors)):
            card = self._frame(grid, bg=C['surface'],
                               highlightbackground=C['border'],
                               highlightthickness=1)
            card.grid(row=0, column=i, padx=6, pady=4, sticky='nsew')
            grid.columnconfigure(i, weight=1)
            self._lbl(card, label, 9, color=C['muted'], bg=C['surface']).pack(anchor='w', padx=12, pady=(10,0))
            self._lbl(card, val, 22, 'bold', color=col, bg=C['surface']).pack(anchor='w', padx=12)
            self._lbl(card, desc, 8, color=C['muted'], bg=C['surface']).pack(anchor='w', padx=12, pady=(0,10))

        self._lbl(self.metrics_inner, "About the model", 13, 'bold').pack(anchor='w', pady=(20,8))
        info = [
            ("Architecture",  "Ensemble — Multinomial Naive Bayes + Logistic Regression"),
            ("Vectorization", "TF-IDF  ·  20 000 features  ·  unigrams & bigrams"),
            ("Dataset",       "NLTK movie_reviews  ·  2 000 reviews  ·  80/20 split"),
            ("Negation fix",  "Kept: not, never, hardly, barely… (improve accuracy)"),
        ]
        for k, v in info:
            row = self._frame(self.metrics_inner, bg=C['surface'],
                              highlightbackground=C['border'],
                              highlightthickness=1)
            row.pack(fill='x', pady=3)
            self._lbl(row, k, 10, 'bold', bg=C['surface']).pack(side='left', padx=12, pady=8)
            self._lbl(row, v, 10, color=C['muted'], bg=C['surface']).pack(side='left', padx=4)

    # ── charts tab ───────────────────────────
    def _build_charts_tab(self, parent):
        f = self._frame(parent)
        self.charts_inner = self._frame(f)
        self.charts_inner.pack(fill='both', expand=True)
        self._lbl(self.charts_inner, "Training model, please wait…",
                  13, color=C['muted']).pack(pady=40)
        return f

    def _populate_charts(self):
        for w in self.charts_inner.winfo_children():
            w.destroy()
        md = self.model_data

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 6), facecolor=C['bg'])
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4)

        PLOT_C = {
            'pos': C['pos'], 'neg': C['neg'],
            'blue': C['blue'], 'purple': C['purple'],
        }

        # confusion matrix
        ax1  = fig.add_subplot(gs[0, 0])
        cm   = confusion_matrix(md['y_test'], md['y_pred'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Neg', 'Pos'])
        disp.plot(ax=ax1, colorbar=False, cmap='Blues')
        ax1.set_title('Confusion Matrix', color=C['text'], fontsize=9)
        ax1.tick_params(colors=C['muted'], labelsize=8)

        # metrics bars
        ax2     = fig.add_subplot(gs[0, 1])
        metrics = ['Accuracy','Precision','Recall','F1']
        values  = [md['acc'], md['precision'], md['recall'], md['f1']]
        bcols   = [PLOT_C['blue'], PLOT_C['pos'], PLOT_C['purple'], PLOT_C['neg']]
        bars    = ax2.bar(metrics, values, color=bcols, width=0.6)
        ax2.set_ylim(0, 1.15)
        ax2.set_facecolor(C['surface'])
        ax2.set_title('Performance Metrics', color=C['text'], fontsize=9)
        ax2.tick_params(colors=C['muted'], labelsize=8)
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02,
                     f'{val:.2f}', ha='center', fontsize=8, color=C['text'])

        # confidence distribution
        ax3      = fig.add_subplot(gs[0, 2])
        y_proba  = np.array(md['y_proba'])
        y_test   = np.array(md['y_test'])
        ax3.hist(y_proba[y_test==1], bins=20, alpha=0.7, color=PLOT_C['pos'], label='Positive')
        ax3.hist(y_proba[y_test==0], bins=20, alpha=0.7, color=PLOT_C['neg'], label='Negative')
        ax3.set_facecolor(C['surface'])
        ax3.set_title('Confidence Distribution', color=C['text'], fontsize=9)
        ax3.tick_params(colors=C['muted'], labelsize=8)
        ax3.legend(fontsize=7)

        # top positive keywords
        ax4        = fig.add_subplot(gs[1, 0])
        lr_model   = md['ensemble'].estimators_[1]
        feat_names = np.array(md['vectorizer'].get_feature_names_out())
        coefs      = lr_model.coef_[0]
        top_pos    = coefs.argsort()[-10:][::-1]
        ax4.barh(feat_names[top_pos][::-1], coefs[top_pos][::-1], color=PLOT_C['pos'])
        ax4.set_facecolor(C['surface'])
        ax4.set_title('Top Positive Words', color=C['text'], fontsize=9)
        ax4.tick_params(colors=C['muted'], labelsize=7)

        # top negative keywords
        ax5     = fig.add_subplot(gs[1, 1])
        top_neg = coefs.argsort()[:10]
        ax5.barh(feat_names[top_neg][::-1], np.abs(coefs[top_neg][::-1]), color=PLOT_C['neg'])
        ax5.set_facecolor(C['surface'])
        ax5.set_title('Top Negative Words', color=C['text'], fontsize=9)
        ax5.tick_params(colors=C['muted'], labelsize=7)

        # review length distribution
        ax6   = fig.add_subplot(gs[1, 2])
        pos_l = [len(t.split()) for t, l in zip(md['texts_clean'], md['labels']) if l == 1]
        neg_l = [len(t.split()) for t, l in zip(md['texts_clean'], md['labels']) if l == 0]
        ax6.hist(pos_l, bins=30, alpha=0.7, color=PLOT_C['pos'], label='Positive')
        ax6.hist(neg_l, bins=30, alpha=0.7, color=PLOT_C['neg'], label='Negative')
        ax6.set_facecolor(C['surface'])
        ax6.set_title('Review Length', color=C['text'], fontsize=9)
        ax6.tick_params(colors=C['muted'], labelsize=8)
        ax6.legend(fontsize=7)

        canvas = FigureCanvasTkAgg(fig, master=self.charts_inner)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    # ── training thread ──────────────────────
    def _train(self):
        def status(msg):
            self.status_var.set(msg)

        data = load_and_train(status)
        self.model_data = data
        self.after(0, self._on_ready)

    def _on_ready(self):
        self._populate_metrics()
        self._populate_charts()
        self.status_var.set(f"Ready  ·  Accuracy {self.model_data['acc']*100:.1f}%")

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app = App()
    app.mainloop()