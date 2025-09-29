from . import utils
import torch
from langdetect import detect
import torch.nn.functional as F
import read_contents.crawl as crawl
from models.model import device
from models.eng_loader import model_eng_tokenizer as tokenizer_eng
from models.kor_loader import model_kor_tokenizer as tokenizer_kor
from models.img_loader import model_img_preprocess
from PIL import Image

def detect_ai_generated_text(text: str, model_eng, model_kor):
    try:
        detected_lang = detect(text)
        if(detected_lang == 'ko'):
            #print(f"Detected language: Korean, {text}")
            prob = detect_ai_generated_text_kor(text, model_kor)
            if prob['max_probability'] is None:
                prob = "error"
            else:
                prob = prob['max_probability']
            return prob
        else:
            #print(f"Detected language: English, {text}")
            return detect_ai_generated_text_eng(text, model_eng)
    except:
        return detect_ai_generated_text_eng(text, model_eng)
    
def detect_ai_generated_text_eng(text : str, model, max_len=128):
    try:
        inputs = tokenizer_eng(
            text, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        ai_probability = probabilities[:, 0].item()
        #print(ai_probability)

        return round(ai_probability, 4)

    except Exception as e:
        #print(f"AI 판별 오류: {e}")
        return None

def detect_ai_generated_text_kor(text: str, model, max_len=258):
    chunk_probabilities = []

    chunk_ids_list = crawl.tokenize_text_kor(text, max_len)

    for chunk_ids in chunk_ids_list:
        if not chunk_ids:
            continue
        
        try:

            tokens = chunk_ids
            sequence_length = len(tokens) + 2
            num_padding = max_len - sequence_length
            padding = [tokenizer_kor.pad_token_id] * num_padding

            final_tokens_list = [tokenizer_kor.bos_token_id] + tokens + [tokenizer_kor.eos_token_id] + padding
            input_ids = torch.tensor(final_tokens_list).unsqueeze(0).to(device)
            
            attention_mask = torch.zeros(max_len, dtype=torch.long)
            attention_mask[:sequence_length] = 1
            attention_mask = attention_mask.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(x=input_ids, attention_mask=attention_mask)
            
            logits = outputs
            probabilities = F.softmax(logits, dim=1)
            ai_probability = probabilities[:, 0].item()
            chunk_probabilities.append(round(ai_probability, 4))
            
        except Exception as e:
            continue

    return utils.format_detection_results(chunk_probabilities)

def detect_ai_generated_image(img: Image, model_img):
    img = model_img_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model_img(img)          # shape: [1, 2]
        probs  = torch.softmax(logits, dim=-1)  # [1,2] – fake/real 확률
        pred_i = torch.argmax(probs, dim=-1).item()  # index 0 or 1
        print(pred_i,probs)
    return pred_i # 0 == fake | 1 == real