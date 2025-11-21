#!/usr/bin/env python3
"""
lab3_app.py

Призначення:
  - Завантажує локальний WAV в S3
  - Запускає AWS Transcribe job для S3 URI
  - Після завершення отримує транскрипт
  - Визначає мову (langdetect)
  - Робить sentiment (NLTK VADER)
  - Шукає фразу та виводить позицію
  - Витягує Named Entities через spaCy (en_core_web_sm)

Використання:
  python lab3_app.py --audio-source lab3.wav --phrase "searched phrase" --s3-bucket my-bucket

Опції:
  --s3-bucket        S3 bucket для тимчасового аплоаду (обов'язково)
  --s3-prefix        префікс у бакеті (за замовчуванням: incoming/)
  --aws-region       AWS region для Transcribe (default: us-east-1)
  --language-code    LanguageCode для Transcribe (наприклад en-US). Якщо не вказаний, використовує 'en-US'
  --keep-s3          не видаляти файл з S3 після завершення
  --keep-local       не видаляти тимчасовий локальний файл
  --timeout          макс час очікування у секундах (default: 600)
"""

import argparse
import os
import sys
import time
import uuid
import json
import tempfile
import boto3
import requests

from langdetect import detect, DetectorFactory, LangDetectException
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import spacy

# Ensure reproducible langdetect
DetectorFactory.seed = 0

SUPPORTED_EXT = {'.wav'}
TRANSCRIBE_POLL_INTERVAL = 5  # seconds

def ensure_nltk_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading NLTK vader_lexicon ...")
        nltk.download('vader_lexicon')

def upload_to_s3(local_path, bucket, key):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, bucket, key)
    except Exception as e:
        raise RuntimeError(f"Failed to upload to S3: {e}")
    return f"s3://{bucket}/{key}"

def start_transcribe_job(s3_bucket, s3_key, job_name, region='us-east-1', language_code='en-US'):
    transcribe = boto3.client('transcribe', region_name=region)
    media_format = os.path.splitext(s3_key)[1].lstrip('.').lower()
    if media_format not in {'wav', 'mp3'}:
        raise ValueError("Transcribe supports only wav/mp3 in this script.")
    media_uri = f"s3://{s3_bucket}/{s3_key}"

    params = {
        'TranscriptionJobName': job_name,
        'LanguageCode': language_code,
        'MediaFormat': media_format,
        'Media': {'MediaFileUri': media_uri},
    }
    transcribe.start_transcription_job(**params)
    return job_name

def wait_for_transcribe(job_name, region='us-east-1', timeout=600, poll_interval=TRANSCRIBE_POLL_INTERVAL):
    transcribe = boto3.client('transcribe', region_name=region)
    start = time.time()
    while True:
        resp = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status = resp['TranscriptionJob']['TranscriptionJobStatus']
        if status == 'COMPLETED':
            return resp['TranscriptionJob']
        if status == 'FAILED':
            raise RuntimeError(f"Transcribe job failed: {resp['TranscriptionJob'].get('FailureReason')}")
        if time.time() - start > timeout:
            raise TimeoutError("Timeout waiting for Transcribe job to complete.")
        time.sleep(poll_interval)

def download_transcript_json(transcript_uri):
    r = requests.get(transcript_uri)
    r.raise_for_status()
    return r.json()

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except LangDetectException:
        return "unknown"

def sentiment_vader(text):
    ensure_nltk_vader()
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    # Use compound score thresholds: >=0.05 positive, <=-0.05 negative, else neutral
    compound = scores['compound']
    if compound >= 0.05:
        label = 'Positive'
    elif compound <= -0.05:
        label = 'Negative'
    else:
        label = 'Neutral'
    return label, scores

def find_phrase(text, phrase):
    if not phrase:
        return False, None
    idx = text.lower().find(phrase.lower())
    if idx == -1:
        return False, None
    # compute word index approx:
    pre = text[:idx]
    char_pos = idx
    # token position (count words before):
    token_pos = len(pre.split())
    return True, {"char_index": char_pos, "word_index": token_pos}

def extract_entities_spacy(text):
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError("spaCy model 'en_core_web_sm' not installed. Run: python -m spacy download en_core_web_sm")
    doc = nlp(text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    return ents

def main():
    parser = argparse.ArgumentParser(description="WAV -> Transcribe -> Lang -> Sentiment -> Phrase -> NER")
    parser.add_argument('--audio-source', required=True, help='Локальний WAV файл')
    parser.add_argument('--phrase', required=False, default='', help='Фраза для пошуку')
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket для тимчасового завантаження')
    parser.add_argument('--s3-prefix', default='incoming/', help='Префікс у бакеті')
    parser.add_argument('--aws-region', default='us-east-1', help='AWS region for Transcribe')
    parser.add_argument('--language-code', default='en-US', help='LanguageCode для Transcribe (default en-US)')
    parser.add_argument('--keep-s3', action='store_true', help='Не видаляти файл з S3')
    parser.add_argument('--keep-local', action='store_true', help='Не видаляти локальний тимчасовий файл')
    parser.add_argument('--timeout', type=int, default=600, help='Макс час очікування Transcribe job (s)')
    args = parser.parse_args()

    audio_path = args.audio_source
    if not os.path.isfile(audio_path):
        print("Файл не знайдено:", audio_path)
        sys.exit(1)

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED_EXT:
        print("Підтримуються лише WAV файли.")
        sys.exit(2)

    # create temporary copy to avoid altering original and to ensure unique key
    tmp_dir = tempfile.gettempdir()
    unique = uuid.uuid4().hex[:8]
    tmp_name = f"lab3_{unique}{ext}"
    tmp_path = os.path.join(tmp_dir, tmp_name)
    import shutil
    shutil.copyfile(audio_path, tmp_path)

    s3_key = os.path.join(args.s3_prefix.rstrip('/'), tmp_name).lstrip('/')
    try:
        print(f"Uploading {tmp_path} -> s3://{args.s3_bucket}/{s3_key} ...")
        upload_to_s3(tmp_path, args.s3_bucket, s3_key)

        job_name = f"lab3_transcribe_{unique}"
        print(f"Starting AWS Transcribe job: {job_name} (language: {args.language_code}) ...")
        start_transcribe_job(args.s3_bucket, s3_key, job_name, region=args.aws_region, language_code=args.language_code)

        print("Waiting for Transcribe to finish (polling)...")
        job_info = wait_for_transcribe(job_name, region=args.aws_region, timeout=args.timeout)
        transcript_uri = job_info['Transcript']['TranscriptFileUri']
        print("Downloading transcript JSON from:", transcript_uri)
        tjson = download_transcript_json(transcript_uri)

        # extract transcript text
        transcript_text = ""
        try:
            transcript_text = tjson['results']['transcripts'][0]['transcript']
        except Exception:
            # fallback: combine items
            items = tjson.get('results', {}).get('items', [])
            transcript_text = " ".join([it.get('alternatives', [{}])[0].get('content', '') for it in items])

        # Language detection
        lang = detect_language(transcript_text)

        # Sentiment
        sentiment_label, sentiment_scores = sentiment_vader(transcript_text)

        # Phrase search
        found, pos = find_phrase(transcript_text, args.phrase)

        # Named entities
        ents = extract_entities_spacy(transcript_text)

        # Output
        print("Transcription:")
        print(transcript_text)
        print("Language:")
        print(lang)
        print("Sentiment:")
        print(sentiment_label)
        print("Scores:", sentiment_scores)
        print("Phrase search:")
        if args.phrase:
            if found:
                print(f"Phrase Found at char index {pos['char_index']}, approx. word index {pos['word_index']}")
            else:
                print("Phrase Not found")
        else:
            print("No phrase provided")

        print("Named entities:")
        if ents:
            for text, label in ents:
                print(f"- {text} ({label})")
        else:
            print("No named entities found")

    finally:
        # cleanup
        if not args.keep_local:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        if not args.keep_s3:
            try:
                s3 = boto3.client('s3')
                s3.delete_object(Bucket=args.s3_bucket, Key=s3_key)
            except Exception:
                pass

if __name__ == '__main__':
    main()
