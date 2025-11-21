#!/usr/bin/env python3
"""
s3_transcribe.py

Використання:
  python s3_transcribe.py --s3-bucket my-bucket --s3-key path/to/file.mp3 --provider aws
  python s3_transcribe.py --s3-bucket my-bucket --s3-key path/to/file.wav --provider deepgram --deepgram-key YOUR_KEY

Параметри:
  --s3-bucket        : назва S3 бакета
  --s3-key           : ключ (шлях) до файлу в бакеті
  --provider         : "aws" або "deepgram" (переважно aws для великих файлів)
  --aws-region       : регіон для AWS Transcribe (default: us-east-1)
  --language-code    : код мови (наприклад, "en-US", "uk-UA") (optional)
  --output           : шлях для збереження тексту транскрипції (default: stdout)
  --deepgram-key     : Deepgram API key (або встановити DEEPGRAM_API_KEY в оточенні)
  --keep-local       : не видаляти тимчасовий локальний файл після виконання
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
from botocore.exceptions import ClientError

SUPPORTED_EXT = {'.mp3', '.wav'}


def guess_media_format(s3_key):
    ext = os.path.splitext(s3_key)[1].lower()
    if ext.startswith('.'):
        ext = ext[1:]
    return ext


def download_s3_to_local(bucket, key, local_path):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket, key, local_path)
    except ClientError as e:
        raise RuntimeError(f"Помилка завантаження з S3: {e}")


# ---------- AWS Transcribe ----------
def transcribe_with_aws(s3_bucket, s3_key, language_code=None, region="us-east-1", wait=True, poll_interval=5):
    """
    Start AWS Transcribe job with media located in S3 (s3://bucket/key).
    Polls until job completes and returns the transcript text and raw response JSON.
    """
    transcribe = boto3.client('transcribe', region_name=region)

    media_format = guess_media_format(s3_key)
    if media_format not in {'mp3', 'wav'}:
        raise ValueError("Підтримуються тільки mp3 та wav для AWS Transcribe.")

    job_name = f"transcribe_{uuid.uuid4().hex[:8]}"
    media_uri = f"s3://{s3_bucket}/{s3_key}"

    params = {
        'TranscriptionJobName': job_name,
        'LanguageCode': language_code or 'en-US',
        'MediaFormat': media_format,
        'Media': {'MediaFileUri': media_uri},
        # якщо бажаєш записувати результат в S3, можна додати OutputBucketName
        # 'OutputBucketName': 'my-output-bucket',
    }

    print(f"Запуск AWS Transcribe job: {job_name} для {media_uri} ...")
    transcribe.start_transcription_job(**params)

    if not wait:
        return {"job_name": job_name, "status": "IN_PROGRESS"}

    # Polling
    while True:
        status_resp = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job = status_resp['TranscriptionJob']
        status = job['TranscriptionJobStatus']
        if status in ('COMPLETED', 'FAILED'):
            break
        print(f"Статус: {status} — очікування {poll_interval}s...")
        time.sleep(poll_interval)

    if status == 'FAILED':
        reason = job.get('FailureReason', 'unknown')
        raise RuntimeError(f"AWS Transcribe job failed: {reason}")

    transcript_uri = job['Transcript']['TranscriptFileUri']
    # Завантажуємо JSON з transcript_uri
    print(f"Завантаження результату транскрипції з {transcript_uri} ...")
    r = requests.get(transcript_uri)
    r.raise_for_status()
    transcript_json = r.json()

    # AWS Transcribe JSON structure: transcript_json['results']['transcripts'][0]['transcript']
    transcript_text = ""
    try:
        transcript_text = transcript_json['results']['transcripts'][0]['transcript']
    except Exception:
        # fallback: try to combine items
        segments = transcript_json.get('results', {}).get('items', [])
        transcript_text = " ".join([seg.get('alternatives', [{}])[0].get('content', '') for seg in segments])

    return {"provider": "aws", "job_name": job_name, "transcript": transcript_text, "raw": transcript_json}


# ---------- Deepgram (ASR) ----------
def transcribe_with_deepgram_localfile(local_path, deepgram_key, language_code=None):
    """
    Upload local file bytes to Deepgram 'listen' endpoint (synchronous).
    Returns transcript text and raw JSON.
    """
    # Deepgram listen API:
    # POST https://api.deepgram.com/v1/listen?model=general&language=...
    # Authorization: Token <API_KEY>
    if not deepgram_key:
        raise ValueError("Deepgram API key is required for Deepgram provider.")

    ext = os.path.splitext(local_path)[1].lower().lstrip('.')
    if ext not in {'mp3', 'wav'}:
        raise ValueError("Deepgram: тільки mp3 та wav підтримуються.")

    url = "https://api.deepgram.com/v1/listen"
    params = {}
    if language_code:
        params['language'] = language_code
    # Optional: choose model param, e.g., 'general', 'general_v2'
    params['model'] = 'general'

    headers = {
        "Authorization": f"Token {deepgram_key}"
    }

    print(f"Відправка файлу {local_path} до Deepgram ...")
    with open(local_path, 'rb') as f:
        resp = requests.post(url, params=params, headers=headers, data=f)
    resp.raise_for_status()
    data = resp.json()

    # Deepgram: results -> channels[0].alternatives[0].transcript
    transcript = ""
    try:
        transcript = data['results']['channels'][0]['alternatives'][0]['transcript']
    except Exception:
        # best-effort flatten
        transcript = json.dumps(data)
    return {"provider": "deepgram", "transcript": transcript, "raw": data}


# ---------- Orchestration CLI ----------
def main():
    parser = argparse.ArgumentParser(description="S3 -> Transcribe (AWS or Deepgram)")
    parser.add_argument('--s3-bucket', required=True, help='S3 bucket name')
    parser.add_argument('--s3-key', required=True, help='S3 key/path to file')
    parser.add_argument('--provider', choices=['aws', 'deepgram'], default='aws', help='Транскриптор')
    parser.add_argument('--aws-region', default='us-east-1', help='AWS region для Transcribe')
    parser.add_argument('--language-code', default=None, help='Код мови (en-US, uk-UA тощо)')
    parser.add_argument('--output', default=None, help='Шлях для збереження транскрипції (txt). Якщо не вказано — виведе в stdout.')
    parser.add_argument('--deepgram-key', default=os.environ.get('DEEPGRAM_API_KEY'), help='Deepgram API key (або DEEPGRAM_API_KEY env var)')
    parser.add_argument('--keep-local', action='store_true', help='Не видаляти тимчасовий локальний файл')
    args = parser.parse_args()

    ext = os.path.splitext(args.s3_key)[1].lower()
    if ext not in SUPPORTED_EXT:
        print("Підтримуються лише mp3 та wav файли.")
        sys.exit(2)

    # Тимчасовий файл
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)  # закриваємо дескриптор, бо boto3 буде писати файл
    try:
        print(f"Завантаження s3://{args.s3_bucket}/{args.s3_key} -> {tmp_path}")
        download_s3_to_local(args.s3_bucket, args.s3_key, tmp_path)

        if args.provider == 'aws':
            result = transcribe_with_aws(args.s3_bucket, args.s3_key, language_code=args.language_code, region=args.aws_region, wait=True)
        else:
            # Deepgram: потребує локального файлу
            result = transcribe_with_deepgram_localfile(tmp_path, args.deepgram_key, language_code=args.language_code)

        transcript_text = result.get('transcript', '')
        raw = result.get('raw')

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            print(f"Транскрипція збережена у {args.output}")
            # також можемо зберегти raw JSON
            raw_path = args.output + '.json'
            with open(raw_path, 'w', encoding='utf-8') as f:
                json.dump(raw, f, ensure_ascii=False, indent=2)
            print(f"Сирі дані (raw) збережені у {raw_path}")
        else:
            print("\n--- TRANSCRIPT ---\n")
            print(transcript_text)
            print("\n--- RAW JSON (скорочено) ---\n")
            try:
                print(json.dumps(raw, ensure_ascii=False, indent=2)[:4000])
            except Exception:
                print(str(raw)[:4000])

    finally:
        if args.keep_local:
            print(f"Тимчасовий файл залишено: {tmp_path}")
        else:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


if __name__ == '__main__':
    main()
