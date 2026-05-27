import random
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

tf.keras.mixed_precision.set_global_policy('mixed_float16')

csv_path = '/content/drive/MyDrive/Colab Notebooks/chatbot_data.csv'
data = pd.read_csv(csv_path).dropna()

questions = data['question'].astype(str).tolist()
answers   = data['answer'].astype(str).tolist()

unique_answers = sorted(list(set(answers)))

answer_to_idx = {ans: idx for idx, ans in enumerate(unique_answers)}
idx_to_answer = {idx: ans for idx, ans in enumerate(unique_answers)}

y = np.array([answer_to_idx[a] for a in answers])
num_classes = len(unique_answers)

print(f"\n답변 클래스 수: {num_classes}")

VOCAB_SIZE = 12000
MAX_LEN = 40

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_LEN
)
vectorizer.adapt(questions)

X = vectorizer(tf.constant(questions)).numpy()

BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(15000)

train_size = int(len(X) * 0.9)

train_ds = dataset.take(train_size)
val_ds   = dataset.skip(train_size)

train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        score = tf.nn.tanh(tf.matmul(inputs, self.W))
        weights = tf.nn.softmax(score, axis=1)
        context = weights * inputs
        return tf.reduce_sum(context, axis=1)

def top_k_sampling(preds, k=5, temperature=0.7):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    top_indices = np.argsort(preds)[-k:]
    top_probs = preds[top_indices]
    top_probs = top_probs / np.sum(top_probs)
    
    chosen = np.random.choice(top_indices, p=top_probs)
    return chosen

inputs = tf.keras.Input(shape=(MAX_LEN,))

x = tf.keras.layers.Embedding(VOCAB_SIZE, 256)(inputs)
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = AttentionLayer()(x)

x = tf.keras.layers.Dense(256)(x)
x = tf.keras.layers.LeakyReLU()(x)
x = tf.keras.layers.Dropout(0.3)(x)

x = tf.keras.layers.Dense(128)(x)
x = tf.keras.layers.LeakyReLU()(x)

outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

model = tf.keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

print("\n[지도학습] 대화 모델 훈련 시작")
model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)
print("지도학습 완료")

embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
question_embeddings = embedding_model.predict(X, batch_size=256, verbose=1)

NUM_TOPICS = 5
print(f"\n[비지도학습] 질문 데이터 군집화 진행 중 (목표 카테고리: {NUM_TOPICS}개)")

kmeans_model = KMeans(n_clusters=NUM_TOPICS, random_state=42, n_init='auto')
question_topics = kmeans_model.fit_predict(question_embeddings)

print("비지도학습 완료: 질문 문맥에 따른 토픽 배정이 끝")

q_table = np.zeros((NUM_TOPICS, 2))
alpha_rl = 0.1
gamma_rl = 0.9
EPSILON = 0.2

def update_q_table(topic, action, is_correct):
    reward = 1.0 if is_correct else -1.0
    
    old_q = q_table[topic, action]
    next_max_q = np.max(q_table[topic, :])
    
    q_table[topic, action] = old_q + alpha_rl * (reward + gamma_rl * next_max_q - old_q)
    print(f"  [RL 보상 반영] 토픽 방 [{topic}] -> 행동 [{action}] f신뢰 점수: {q_table[topic, action]:.4f}")

def chatbot_predict_hybrid(user_input, epsilon=EPSILON):
    vec_input = vectorizer(tf.constant([user_input]))
    user_embedding = embedding_model.predict(vec_input, verbose=0)
    
    current_topic = kmeans_model.predict(user_embedding)[0]
    
    if np.random.rand() < epsilon:
        action = np.random.choice([0, 1])
        reason = "강화학습 무작위 실험 모드(Exploration)"
    else:
        action = np.argmax(q_table[current_topic, :])
        reason = "강화학습 최적 점수판 모드(Exploitation)"
        
    pred = model.predict(vec_input, verbose=0)[0]
    pred_idx = top_k_sampling(pred, k=5, temperature=0.7)
    ai_answer = idx_to_answer[pred_idx]
    confidence = float(pred[pred_idx])
    
    sims = cosine_similarity(user_embedding, question_embeddings)[0]
    best_idx = np.argmax(sims)
    similarity = sims[best_idx]
    retrieval_answer = answers[best_idx]
    retrieval_question = questions[best_idx]
    
    if similarity > 0.85:
        final_answer = retrieval_answer
        mode_str = "실시간 피드백 유연 반영 (유사 문장 라우팅)"
    elif action == 0:
        final_answer = retrieval_answer
        mode_str = "검색기반 모드"
    else:
        final_answer = ai_answer
        mode_str = "AI 생성 모드"
        
    if action == 1 and confidence < 0.20 and similarity <= 0.85:
        final_answer = "잘 모르겠어요."
        mode_str = "불확실 방어 메커니즘"
        
    print("\n==============================================")
    print(f" AI 답변: {final_answer}")
    print("==============================================")
    print(f" 시스템 판단 가이드:")
    print(f" 인식된 문맥 카테고리: {current_topic}번 토픽방 (비지도학습)")
    print(f" 라우터 의사결정 방식: {reason}")
    print(f" 최종 선택된 출력 수단: {mode_str}")
    print(f" 참고 지표 - AI 확신도: {confidence*100:.2f}% | 검색 유사도: {similarity*100:.2f}%")
    print(f" 가장 유사한 기존 질문: '{retrieval_question}'")
    
    return final_answer, current_topic, action

def online_learning(user_input, correct_answer):
    global questions, answers, question_embeddings
    
    if correct_answer not in answer_to_idx:
        new_idx = len(answer_to_idx)
        answer_to_idx[correct_answer] = new_idx
        idx_to_answer[new_idx] = correct_answer
        
    questions.append(user_input)
    answers.append(correct_answer)
    
    x_new = vectorizer(tf.constant([user_input]))
    y_new = np.array([answer_to_idx[correct_answer]])
    
    model.fit(x_new, y_new, epochs=4, verbose=0)
    
    new_embed = embedding_model.predict(x_new, verbose=0)
    question_embeddings = np.vstack([question_embeddings, new_embed])
    print("[온라인 지도학습] 과적합 방지형 미세조정(Epochs=4) 및 DB 갱신")

print("대화를 종료하려면 '종료'를 입력하세요.")
print("==================================================\n")

while True:
    user_input = input("\n나: ").strip()
    
    if user_input == '종료':
        break
    if not user_input:
        continue
        
    answer, topic, action = chatbot_predict_hybrid(user_input)
    
    feedback = input("\n만족스러운 답변인가요? (Y/N): ").strip().upper()
    is_correct = (feedback == 'Y')
    
    update_q_table(topic, action, is_correct)
    
    if not is_correct:
        correct = input("올바른 답: ").strip()
        if correct:
            online_learning(user_input, correct)

print("\n종료되었습니다.")
